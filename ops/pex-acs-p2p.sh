#!/usr/bin/env bash
# pex-acs-p2p.sh - clear ACS P2P redirect on the PCIe switch ports between your GPUs, so GPU<->GPU peer
# traffic stays inside the switches instead of going up to the root port and back.
#
# Why: when an IOMMU is present - even with iommu=pt - Linux sets ACS P2P Request Redirect and P2P
# Completion Redirect on every switch downstream port that has an ACS capability. Measured on our box (two
# PLX PEX8796 GPU boards under a Broadcom PEX880xx Gen4 switch): with redirect on, board-to-board P2P took
# 1.79 us one way, slower than the 1.58 us it took through the CPU root complex before the switch went in;
# clearing it gave 1.25 us and raised SM P2P reads from 7.1 to 9.4 GB/s. Copy-engine bandwidth stayed at
# 13.2 GB/s: each board's Gen3 x16 uplink is the limit either way.
#
# It clears only RR|CR|EC (0x002c), the bits the kernel's pci=disable_acs_redir= clears; Source Validation
# and Upstream Forwarding stay on. That removes the isolation ACS gives the devices behind those ports,
# which matters if you pass devices through to VMs.
#
# Which ports: every downstream port of every switch on the path between two of your GPUs.
#   lspci -tv                                                 the tree: which switch port sits above which GPU
#   lspci -vvv -s <BDF> | grep -A2 'Access Control Services'  ACSCtl shows ReqRedir+ CmpltRedir+ while on
# Configure one of these, in the environment or in /etc/default/pex-acs-p2p for the systemd unit:
#   ACS_PORTS="0000:41:08.0 0000:41:10.0"   explicit downstream-port BDFs. BDFs can renumber after BIOS,
#                                            firmware or slot changes: re-check them after any.
#   ACS_VENDEVS="10b5:8796 1000:c010"        vendor:device of your switches (lspci -nn); every port of those
#                                            devices with an ACS capability is cleared. This default names the
#                                            PLX PEX8796 and Broadcom PEX880xx on our box; set your own.
# A port that goes through pci_restore_state (AER recovery, runtime PM) gets the kernel default back, so
# `status` exits 1 if any redirect bit is set again. Empty switch branches runtime-suspend and every resume
# (a config read is enough) restores the default, so `apply` pins each port it clears in D0.
#
#   apply    clear the redirect bits; the first apply per boot saves the previous words in $STATE
#   restore  put the saved words back (the kernel default RR|CR when there is no state file)
#   status   print each port's ACS control word; exit 1 if any port still redirects
# Exit 3: no port matched ACS_PORTS / ACS_VENDEVS. Needs root and pciutils (lspci, setpci).
set -euo pipefail

VENDEVS=${ACS_VENDEVS:-"10b5:8796 1000:c010"}
REDIR=$((0x002c))   # RR 0x04 | CR 0x08 | EC 0x20
STATE=${ACS_STATE:-/run/pex-acs-p2p.orig}
SYSFS=${ACS_SYSFS:-/sys}   # overridable for testing

ports() {
  local v d
  if [ -n "${ACS_PORTS:-}" ]; then
    for d in $ACS_PORTS; do
      [[ "$d" == *:*:* ]] || d="0000:$d"   # sysfs paths need the PCI domain
      if setpci -s "$d" ECAP_ACS+6.w >/dev/null 2>&1; then
        echo "$d"
      else
        echo "pex-acs-p2p: $d has no ACS capability or does not exist; skipped" >&2
      fi
    done
    return 0
  fi
  for v in $VENDEVS; do
    for d in $(lspci -Dn -d "$v" | awk '{print $1}'); do
      if setpci -s "$d" ECAP_ACS+6.w >/dev/null 2>&1; then echo "$d"; fi
    done
  done
}

acs() { echo $((0x$(setpci -s "$1" ECAP_ACS+6.w))); }

mapfile -t PORTS < <(ports)
if [ "${#PORTS[@]}" -eq 0 ] && [ "${1:-status}" != restore ]; then
  echo "pex-acs-p2p: no ACS-capable switch port matched ACS_PORTS='${ACS_PORTS:-}' ACS_VENDEVS='$VENDEVS'" >&2
  exit 3
fi

case "${1:-status}" in
  apply)
    keep=0; [ -s "$STATE" ] && keep=1
    for d in "${PORTS[@]}"; do
      echo on >"$SYSFS/bus/pci/devices/$d/power/control"
      cur=$(acs "$d"); new=$((cur & ~REDIR))
      [ "$keep" = 1 ] || printf '%s %04x\n' "$d" "$cur" >>"$STATE"
      setpci -s "$d" ECAP_ACS+6.w="$(printf '%04x' "$new")"
      printf '%s acs %04x -> %04x\n' "$d" "$cur" "$new"
    done ;;
  restore)
    if [ -s "$STATE" ]; then
      while read -r d w; do
        setpci -s "$d" ECAP_ACS+6.w="$w"; echo "$d acs -> $w"
      done <"$STATE"
      rm -f "$STATE"
    else
      for d in "${PORTS[@]}"; do
        setpci -s "$d" ECAP_ACS+6.w="$(printf '%04x' $(($(acs "$d") | 0x000c)))"
      done
    fi
    for d in "${PORTS[@]}"; do echo auto >"$SYSFS/bus/pci/devices/$d/power/control"; done ;;
  status)
    rc=0
    for d in "${PORTS[@]}"; do
      w=$(acs "$d")
      if ((w & REDIR)); then s=REDIRECT; rc=1; else s=direct; fi
      printf '%s acs %04x %s\n' "$d" "$w" "$s"
    done
    exit "$rc" ;;
  *) echo "usage: $0 apply|restore|status" >&2; exit 2 ;;
esac
