import pandas as pd
from termcolor import colored

# this code calculates some statistics that we are going to use to apply effective min-max scaling to the values 

frames_df = pd.read_csv(r"C:\PROJECT_drowsiness\processed_videos\frames_data.csv")

# calculate ear and mar range for every video independently
video_ranges_df = frames_df.groupby("Video Identifier").agg(
    ear_range = ("EAR", lambda ear: ear.max() - ear.min()),
    mar_range = ("MAR", lambda mar: mar.max() - mar.min())
)

print(colored("Below you can see the first 5 lines of the new df...", on_color = "on_green", attrs = ["bold", "italic"]))

print("\n")
print(video_ranges_df.head())
print("\n")

# 10th percentile means 90% of the ranges is above that value
print(f"Median off the EAR ranges: {video_ranges_df["ear_range"].median():.3f} | 10th percentile of the EAR ranges: {video_ranges_df["ear_range"].quantile(0.10):.3f}")
print(f"Median of the MAR ranges: {video_ranges_df["mar_range"].median():.3f} | 10th percentile of the MAR ranges: {video_ranges_df["mar_range"].quantile(0.10):.3f}")