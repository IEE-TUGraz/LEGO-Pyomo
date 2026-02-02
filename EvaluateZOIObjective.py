import glob
import re
import time

import cloudpickle

from InOutModule.printer import Printer
from LEGO import LEGOUtilities

printer = Printer.getInstance()


def extract_parameters(filename):
    """Extract dcBuffer, tpBuffer, zone, demand, and pmax values from filename."""
    match = re.search(r'-zoi(?P<zone>[^-]+)-.*?dcBuffer(?P<dc>\d+)-tpBuffer(?P<tp>\d+)(?:-demand(?P<demand>\d+(?:\.\d+)?))?(?:-pmax(?P<pmax>\d+(?:\.\d+)?))?', filename)
    if match:
        demand = float(match.group('demand')) if match.group('demand') else 1.0
        pmax = float(match.group('pmax')) if match.group('pmax') else 1.0
        return int(match.group('dc')), int(match.group('tp')), match.group('zone'), demand, pmax
    return None, None, None, 1.0, 1.0


def main():
    # Find all pickle files in the current directory
    pickle_files = glob.glob("*.pkl")

    if not pickle_files:
        printer.warning("No pickle files found in current directory")
        return

    printer.information(f"Found {len(pickle_files)} pickle file(s)")

    # Process each pickle file and group by base identifier (everything except zone)
    results = []
    file_groups = {}  # Key: base_identifier, Value: list of (pkl_file, dc_buffer, tp_buffer, zone, zoi_value, model)

    for pkl_file in sorted(pickle_files):
        printer.information(f"\nProcessing '{pkl_file}'...")

        try:
            # Load the LEGO Pyomo model
            start_time = time.time()
            with open(pkl_file, mode='rb') as file:
                model = cloudpickle.load(file)
            load_time = time.time() - start_time
            printer.information(f"  Loaded in {load_time:.2f} seconds")

            # Calculate the ZOI objective function
            calc_start_time = time.time()
            zoi_expr, zoi_value = LEGOUtilities.evaluate_zoi_objective(model, line_filter="both")
            calc_time = time.time() - calc_start_time
            printer.information(f"  ZOI objective calculated in {calc_time:.2f} seconds")

            # Extract parameters from filename
            dc_buffer, tp_buffer, zone, demand, pmax = extract_parameters(pkl_file)

            # Create base identifier by removing zone from filename
            base_identifier = re.sub(r'-zoi[^-]+', '-zoi', pkl_file)

            printer.success(f"  ZOI Objective: {zoi_value:.2f}")
            results.append((pkl_file, dc_buffer, tp_buffer, zone, demand, pmax, zoi_value))

            # Group files for comparison
            if base_identifier not in file_groups:
                file_groups[base_identifier] = []
            file_groups[base_identifier].append((pkl_file, dc_buffer, tp_buffer, zone, demand, pmax, zoi_value, model))

        except Exception as e:
            printer.error(f"  Failed to process '{pkl_file}': {e}")

    # Print summary
    if results:
        # Calculate the maximum filename length for proper alignment
        max_filename_len = max(len(pkl_file) for pkl_file, _, _, _, _, _, _ in results)
        # Ensure minimum width for readability
        filename_width = max(max_filename_len, len("Filename"))
        # Calculate total table width
        table_width = filename_width + 2 + 8 + 8 + 8 + 8 + 14  # 2 for spacing, rest for columns

        printer.information("\n" + "=" * table_width)
        printer.information("Summary of ZOI Objective Values:")
        printer.information("=" * table_width)
        printer.information(f"  {'Filename':<{filename_width}s} {'DC-Buf':>8s} {'TP-Buf':>8s} {'Demand':>8s} {'PMax':>8s} {'ZOI Objective':>14s}")
        printer.information("-" * table_width)
        for pkl_file, dc_buffer, tp_buffer, zone, demand, pmax, zoi_value in results:
            dc_str = str(dc_buffer) if dc_buffer is not None else "N/A"
            tp_str = str(tp_buffer) if tp_buffer is not None else "N/A"
            demand_str = f"{demand:.1f}" if demand is not None else "N/A"
            pmax_str = f"{pmax:.1f}" if pmax is not None else "N/A"
            printer.information(f"  {pkl_file:<{filename_width}s} {dc_str:>8s} {tp_str:>8s} {demand_str:>8s} {pmax_str:>8s} {zoi_value:>14.2f}")

    # Compare zoiNone models with other zones in their groups
    for base_identifier, group in file_groups.items():
        # Find the zoiNone model in this group
        zoi_none_entry = next((entry for entry in group if entry[3] == "None"), None)

        if zoi_none_entry is None:
            continue

        zoi_none_file, _, _, _, _, _, zoi_none_original_value, zoi_none_model = zoi_none_entry

        # Get other zones in this group
        other_zones = [entry for entry in group if entry[3] != "None"]

        if not other_zones:
            continue

        printer.information("\n" + "=" * 120)
        printer.information(f"ZOI-None Model Comparisons for: {base_identifier}")
        printer.information("=" * 120)
        printer.information(f"  {'Zone':<20s} {'ZOI Objective (zoiNone model)':>30s} {'ZOI Objective (original)':>30s} {'Difference':>15s} {'Rel. Diff (%)':>15s}")
        printer.information("-" * 120)

        sum_of_zone_objectives = 0.0

        for pkl_file, dc_buffer, tp_buffer, zone, demand, pmax, original_zoi_value, zone_model in other_zones:
            try:
                # Extract zoi_i from the zone model
                zone_zoi_i = list(zone_model.zoi_i)

                # Clear and update zoi_i in the zoiNone model
                zoi_none_model.zoi_i.clear()
                zoi_none_model.zoi_i.construct()
                for bus in zone_zoi_i:
                    zoi_none_model.zoi_i.add(bus)

                # Recalculate ZOI objective with updated zoi_i
                _, zoi_none_value = LEGOUtilities.evaluate_zoi_objective(zoi_none_model, line_filter="both")

                difference = zoi_none_value - original_zoi_value
                rel_diff_pct = (difference / zoi_none_value * 100) if zoi_none_value != 0 else 0.0
                sum_of_zone_objectives += zoi_none_value
                printer.information(f"  {zone:<20s} {zoi_none_value:>30.2f} {original_zoi_value:>30.2f} {difference:>15.2f} {rel_diff_pct:>14.1f}%")

            except Exception as e:
                printer.error(f"  Failed to compare zone {zone}: {e}")

        # Safety check: sum of individual zone objectives should equal original zoiNone objective
        printer.information("-" * 120)
        printer.information(f"  {'SAFETY CHECK':<20s} {'Sum of zone objectives':>30s} {'Original zoiNone objective':>30s} {'Difference':>15s} {'Rel. Diff (%)':>15s}")
        safety_difference = sum_of_zone_objectives - zoi_none_original_value
        safety_rel_diff_pct = (safety_difference / sum_of_zone_objectives * 100) if sum_of_zone_objectives != 0 else 0.0
        tolerance = 0.01  # Allow small numerical differences
        if abs(safety_difference) < tolerance:
            printer.success(f"  {'[PASSED]':<20s} {sum_of_zone_objectives:>30.2f} {zoi_none_original_value:>30.2f} {safety_difference:>15.2f} {safety_rel_diff_pct:>14.1f}%")
        else:
            printer.error(f"  {'[FAILED]':<20s} {sum_of_zone_objectives:>30.2f} {zoi_none_original_value:>30.2f} {safety_difference:>15.2f} {safety_rel_diff_pct:>14.1f}%")


if __name__ == "__main__":
    main()
