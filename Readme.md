# Reading Salmon Fish History from Scale Images Using Deep Transfer Learning
## This project is a part of EMJMD in DEPEND at the University of St Andrews
## This system is built By Ashraf Sinan, under supervision of Prof. Kasim Terzic.
## This project was implemented in partnership with Marine Science Sctoland (MSS)

The system use python3, Tensorflow, OpenCV, PIL, NUMPY, pandas, and matplot lib

1- To run the expirement result open the terminal and change directory to the system folder then execute "python3 evaluation.py" this will show the different models accuracy and confusion matrix
2- To try the profile extractor execute the "python3 profile_extractor.py" this will extract the ciculi profiles, plot them, and save them into the profiles_plot folder
3- To use the generator execute "Scale_Generator_v3.py" this will generate 1000 image into the generated image folder
4- The generated images are not included in the folder because their size is big but they can be generated as listed above and available on the university server(Thanos)
5- The system use U-NET convolutional Neural networks to segment the salmon scale texture and read their history


Note: 
The expirement on different devices show a slightly different results
