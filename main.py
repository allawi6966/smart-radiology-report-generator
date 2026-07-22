import os
import medgemma2_grounded as m
import test_harness as t
if __name__ == "__main__":
    while (True) :

        frontal_path = input ("*** Insert the frontal xray image path ***")
        if os.path.exists(frontal_path) : break
        print ("pls enter a valid and correct frontal image path \n")
    while (True) :
    
            lateral_path = input ("*** Insert the lateral xray image path ***")
            if os.path.exists(lateral_path) : break
            print ("pls enter a valid and correct lateral image path \n ")
    output_dir = "output"
    report_path = os.path.join(output_dir, "reports_grounded.json")
    annotated_path = os.path.join(output_dir, "annotated.png")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model...")
    model, processor = m.load_model()

    print(f"Generating grounded structured report for: {frontal_path}")
    report = m.generate_structured_report(
        model, processor,
        frontal_path,
        lateral_image_path=lateral_path,
    )

    if not m.validate_grounded(report):
        print("⚠️  Missing bounding boxes on one or more findings — retrying once.")
        report = m.generate_structured_report(
            model, processor, frontal_path,
            lateral_image_path=lateral_path,
            max_new_tokens=1024,
        )

    report = m.clean_report(report)

    m.print_structured_report(report)
    m.save_report_json(report, report_path)

    if report.get("findings"):
        m.draw_bounding_boxes(frontal_path, report, annotated_path)
    else:
        print("No findings to draw.")
    t.main()