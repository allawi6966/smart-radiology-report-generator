import medgemma2 as M
import csv
import json
import eval as e


if __name__ == "__main__":
    DATA_PATH="data/images/images_normalized"
    PROJECTIONS_PATH="data/indiana_projections.csv"
    REAL_REPORST='data/indiana_reports.csv'
    FRONTAL_IMAGE_PATH = "1.png"    # <-- change this to your frontal image
    LATERAL_IMAGE_PATH = "2.png"    # <-- change this to your lateral image, or set to None
    INDICATION = ""                 # <-- optional, e.g. "cough, rule out pneumonia"
    FINAL_REPORSTS='final.csv'
    y=[',']
    model, processor = M.load_model()
    with open(PROJECTIONS_PATH,'r') as p :
        x=csv.DictReader(p)
        for row in x :
            if (y[0]==row['uid']):
                y.append(DATA_PATH+'/'+row['filename'])
                print(y)
                report = M.generate_report(
                    model, processor, y[1], y[2], INDICATION
                )

                print ("***new report created for uid : "+ y[0]+"\n")
                print (report+'\n')
                entry={'uid':y[0],'report':report}
                with open ('final.json','a') as file :
                    file.write(json.dumps(entry) + '\n')
                    
            else:
                if (len(y)<3):
                    print("one of the lateral or the frontal images is missing ! skipping to the next client ...")
                y=[row['uid'],DATA_PATH+'/'+row['filename']]


                   
    e.main()
