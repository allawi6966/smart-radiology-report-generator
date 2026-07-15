# smart-radiology-report-generator
Due to some accuracy issues while using MAIRA-2, I switched to using MedGemma so that it could generate valid reports. The current output is free-text format.

Example:
General Impression:
The images show a frontal and lateral view of the chest. The patient appears to be a female ...

The structured report generation is still a work in progress.

there are some concerns about the speed of the report generation , my machine , which has 6gb of vram , needs about 3 to 4 minutes to generate a single report.
for that reason the 'final.json' contains only 33 reports.

For evaluation, I used BLEU and ROUGE (ROUGE-1/2/L) to measure lexical overlap between MedGemma's generated reports and the ground-truth report.
The evaluation results are present in the eval_results.csv .

update : 
*** added structured report generation generation ***
  run the medgemma2_structured.py and change the the lateral image and frontal image path , and also
  the output path if u want .

  the output should be like the one appearing in reports.json in the output directory.
  the output should be something like this :
    
"findings": [
    {
      "id": "finding_001",
      "finding_name": "Cardiomegaly",
      "location": "Heart",
      "severity": "mild",
      "description": "The cardiac silhouette appears mildly enlarged.",
      "confidence": 0.9
    },
    {
      "id": "finding_002",
      "finding_name": "Normal Lung Fields",
      "location": "Both lung fields",
      "severity": "normal",
      "description": "No focal consolidation, pleural effusion, or pneumothorax is identified.",
      "confidence": 1
    }
  ],
  "impression": "The heart is mildly enlarged. The lungs are clear. No acute cardiopulmonary abnormalities are identified.",
  "abnormal": false
}


