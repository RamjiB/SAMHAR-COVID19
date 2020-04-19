# SAMHAR-COVID19
## Prototype Website: https://xraycovid19.com/
#### The diagnosis of COVID-19 is currently performed by a reverse-transcription polymerase chain reaction (RT-PCR) test on blood samples of patients. Though the gold standard test for COVID-19 makes a progressive increase in the number of tests taken as shown in the image below, the increase rate can be further improved by a faster diagnosis using CT scans/X-Rays.

#### Data: X-Ray image data
#### Model: A basic CNN model
#### Website Host: Render
#### Database: AWS
#### Primary result: 
      - The model can successfully say 20 X-rays are COVID positive among 24 X-rays
      - Among the missed four X-rays, two X-rays were classified as Bacterial infection and the other two X-rays as Viral infection.
      - Likewise, Three normal X-rays were classified as COVID-19; Three bacteria-infected X-rays were classified as COVID-19. 
      - None of the viral infected X-rays are falsely predicted as COVID-19 
