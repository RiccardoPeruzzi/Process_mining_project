import pm4py
import pandas as pd

print("-------- LOADING LOG --------")

log = pm4py.read_xes("../Dataset/HospitalBilling.xes")

print("\n-------- LOG STATISTICS --------")

# Number of cases and events
num_cases = len(log)
num_events = sum(len(trace) for trace in log)
print(f"Traces: {num_cases}")
print(f"Events: {num_events}")

# Activities
activities = pm4py.get_event_attribute_values(log, "concept:name")
print(f"\nNumber of distinct activities: {len(activities)}")
print("Activity occurrences:")
for act, count in sorted(activities.items(), key=lambda x: -x[1]):
    print(f"  {act}: {count}")

# Start and end activities
start_acts = pm4py.get_start_activities(log)
end_acts   = pm4py.get_end_activities(log)
print(f"\nStart activities: {len(start_acts)}")
print("Start activities:", start_acts)
print(f"\nNumber of end activities: {len(end_acts)}")
print("End activities:", end_acts)

# Variants
variants = pm4py.get_variants(log)
print(f"\nNumber of variants: {len(variants)}")
