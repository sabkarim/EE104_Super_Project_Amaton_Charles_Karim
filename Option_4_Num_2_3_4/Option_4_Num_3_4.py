
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

#################################################[RESOURCES]#################################################
# CIFAR-10 tutorial and database                                                                            #
# https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/   #
# https://www.cs.toronto.edu/~kriz/cifar.html                                                               #
#############################################################################################################

# load and prepare the image
def load_image(filename):
    img = load_img(filename, target_size=(32, 32))                 # Load the target image
    img = img_to_array(img)                                        # Convert the image to an array
    img = img.reshape(1, 32, 32, 3)                                # Reshape into a 3-channel sample
    img = img.astype('float32')                                    # Prepare pixel data
    img = img / 255.0
    
    return img

# This function allows multiple images to be processed 
def multiprocessing():
    while(True):  
        run_example()
        plt.show(block=True)
        
        exit_statement=input("Enter 'q' if you would like to quit or press any key to continue.")
        
        if (exit_statement.lower()=='q'):
            break
        else:
            print("Program will continue")
            
    return None

# This function calculates the percent match of each image for each category
def getPercentage(prediction_class,index):
    
    i = index
    percentage = round(prediction_class[0][i]*100,3)
                       
    return percentage

# This function loads the chosen image and makes a prediction of what the image is
def run_example():
    print("Please make a selection on the menu below\n")       # Displays the menu and takes user input
    choose = input('''\t\tMenu:                                
              -----------------------------------
              1: Airplane_1_database_processed
              2: Airplane_2_database_processed
              3: Automobile_1_database_processed
              4: Automobile_2_database_processed
              5: Bird_1_database_processed
              6: Bird_2_database_processed
              7: Cat_1_database_processed
              8: Cat_2_database_processed
              9: Deer_1_database_processed
             10: Deer_2_database_processed
             11: Dog_1_database_processed
             12: Dog_2_database_processed
             13: Frog_1_database_processed
             14: Frog_2_database_processed
             15: Horse_1_database_processed
             16: Horse_2_database_processed
             17: Ship-1_database_processed
             18: Ship_2_database_processed
             19: Truck_1_database_processed
             20: Truck_2_database_processed
            -------------------------------------
            Enter here:''')            

    try:                                                           # try block to check valid user input
        options=int(choose)        
        if (options == 1):
            image_import ='Airplane_1_database_processed'
        if (options == 2):
            image_import ='Airplane_2_database_processed'    
        if (options == 3):
            image_import ='Automobile_1_database_processed'
        if (options == 4):
            image_import ='Automobile_2_database_processed'           
        if (options == 5):
            image_import ='Bird_1_database_processed'
        if (options == 6):
            image_import ='Bird_2_database_processed'    
        if (options == 7):
            image_import ='Cat_1_database_processed'
        if (options == 8):
            image_import ='Cat_2_database_processed'
        if (options == 9):
            image_import ='Deer_1_database_processed'
        if (options == 10):
            image_import ='Deer_2_database_processed'    
        if (options == 11):
            image_import ='Dog_1_database_processed'
        if (options == 12):
            image_import ='Dog_2_database_processed'           
        if (options == 13):
            image_import ='Frog_1_database_processed'
        if (options == 14):
            image_import ='Frog_2_database_processed'    
        if (options == 15):
            image_import ='Horse_1_database_processed'
        if (options == 16):
            image_import ='Horse_2_database_processed'
        if (options == 17):
            image_import ='Ship_1_database_processed'
        if (options == 18):
            image_import ='Ship_2_database_processed'    
        if (options == 19):
            image_import ='Truck_1_database_processed'
        if (options == 20):
            image_import='Truck_2_database_processed'         
        
        image = load_image(''+image_import+'.png')                  # Call the load_image function 
        model = load_model('final_model.h5')                        # Load the model using keras
        result = model.predict_classes(image)                       # Predict the class the image falls into
        plt.imshow(mpimg.imread(''+image_import+'.png'))
        prediction_class = model.predict_proba(image)               # Calculate the class probability for the object
        
# Call the getPercentage function for each category
######################################################################################################################
        airplane_stats = getPercentage(prediction_class,0)       
        automobile_stats = getPercentage(prediction_class,1)
        bird_stats = getPercentage(prediction_class,2)
        cat_stats = getPercentage(prediction_class,3)
        deer_stats = getPercentage(prediction_class,4)
        dog_stats = getPercentage(prediction_class,5)
        frog_stats = getPercentage(prediction_class,6)
        horse_stats = getPercentage(prediction_class,7)
        ship_stats = getPercentage(prediction_class,8)
        truck_stats = getPercentage(prediction_class,9)
######################################################################################################################
        
# Display the results in the console window
######################################################################################################################
        print("\nImage:",image_import,"\nMaking Prediction....\n")
        print("Image Prediction:")
        print(f"Airplane: {airplane_stats}%")
        print(f"Automobile: {automobile_stats}%")
        print(f"Bird: {bird_stats}%")
        print(f"Cat: {cat_stats}%")
        print(f"Deer: {deer_stats}%")
        print(f"Dog: {dog_stats}%")
        print(f"Frog: {frog_stats}%")
        print(f"Horse: {horse_stats}%")
        print(f"Ship: {ship_stats }%")
        print(f"Truck: {truck_stats}%")
######################################################################################################################
        
        print("\nClasses: ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']")
        print("Class:",result[0],'\n')
    
        placeMarker = int(result[0])
        
        if (placeMarker == 0):
            print("This is an airplane")        
        if (placeMarker == 1):
            print("This is an automobile")       
        if (placeMarker == 2):
            print("This is a bird")       
        if (placeMarker == 3):
            print("This is a cat.")        
        if (placeMarker == 4):
            print("This is a deer.")        
        if (placeMarker == 5):
            print("This is a dog.")       
        if (placeMarker == 6):
            print("This is a frog.")        
        if (placeMarker == 7):
            print("This is a horse.")        
        if (placeMarker == 8):
            print("This is a ship.")        
        if (placeMarker == 9):
            print("This is a truck.")

    except ValueError:  
        print("Error: a non-integer value was entered.\n"
              "Please only enter a valid integer between 1 and 20")
    except UnboundLocalError:
        print("Error: This integer is out of the given range.\n"
              "Please only enter a valid integer between 1 and 20")
        
    return None
def main():
     
    user_choice = input("Would you like to enable multi-image processing? Type: 'y', or type 'n' for single image processing:")

    if(user_choice.lower()=='y'):
        multiprocessing()    
    elif(user_choice.lower()=='n'):
        run_example()
    else:
        print("Please make a valid selection")
        main()
        
if __name__ == '__main__':
    main()