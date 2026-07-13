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
