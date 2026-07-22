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


*** added grounded reports :
now the reports show where the anomaly is and also the xray images will contain boxes as shown in 
output/annotated.png
<img width="2500" height="2048" alt="annotated" src="https://github.com/user-attachments/assets/550f8575-7786-4d55-a472-365fcf26e35d" />

***  added grounded verification :  check verification report in output directory , it also verifies the boxes on radiology images : (check verification_visualization.png )

<img width="2500" height="2048" alt="verification_visualization" src="https://github.com/user-attachments/assets/0dd32506-465d-494a-89ec-8eb89a92363e" />


*** Instructions before usage  ***

just download the code and run pip install -r requirements.txt , then run the main file , make sure to enter the correct lateral and frontal xray images path .


Do this for now , until i create an gui .


the report , report's verification and grounding boxes images will be saved in output/ . 