import csv

LIST_FILE = "/gpfs/helios/home/dwenger/back_of_car_filtered.txt"
INPUT_CSV = "/gpfs/helios/home/dwenger/detections.csv"
OUTPUT_CSV = "/gpfs/helios/home/dwenger/detections_with_crop_path.csv"

def add_crop_paths():
    """Add crop_path column by matching rows in order."""
    print(f"Reading crops from: {LIST_FILE}")
    print(f"Reading CSV from: {INPUT_CSV}")
    print(f"Writing to: {OUTPUT_CSV}")
    
    rows_processed = 0
    
    with open(LIST_FILE, "r") as crop_file, \
         open(INPUT_CSV, "r") as inf, \
         open(OUTPUT_CSV, "w", newline="") as outf:
        
        reader = csv.DictReader(inf)
        
        # Add crop_path to fieldnames
        fieldnames = reader.fieldnames + ["crop_path"]
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()
        
        # Iterate through both files in parallel
        for crop_line, csv_row in zip(crop_file, reader):
            crop_path = crop_line.strip()
            csv_row["crop_path"] = crop_path
            writer.writerow(csv_row)
            
            rows_processed += 1
            if rows_processed % 100000 == 0:
                print(f"  Processed {rows_processed:,} rows...")
    
    print(f"\nDone! Processed {rows_processed:,} rows")
    print(f"Output saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    add_crop_paths()
