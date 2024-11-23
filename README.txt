The normal progression of this code is to:

THE CompCars FOLDER WITH ALL THE ORIGINAL STUFF IS SUPPOSED TO BE AT THE SAME LEVEL OF THE FOLDER THAT CONTAINS THESE FILES.
THIS IS BECAUSE THIS FOLDER IS CONNECTED WITH GITHUB AND THE CONTENT OF COMPCARS SHOULD NOT BE TRACKED OR UPLOADED.

- Run crop_images.py once to create the cropped_image folder with all the tailored crop_images
- Run create_train_test_splits.py one or more times, based on if you want only one split or different splits for different reasons or runs
- Run train_model.py with the wanted parameters
- Run test_model.py with the same splits_folder used during training

FOR NOW THE RESULTS ARE NOT SAVED ANYWHERE, JUST DISPLAYED IN COMMAND LINE. ONLY THE WHOLE MODEL IS SAVED