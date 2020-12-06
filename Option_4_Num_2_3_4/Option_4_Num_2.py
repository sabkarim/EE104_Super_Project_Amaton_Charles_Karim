
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image as imageprocessing

def main():
    #This program converts input .png images to 32x32 to create our CIFAR-10 database of images
    
    #Put the image name here without '.png'
    originalimage='Bird_2'
    
    #Convert to 32x32
    plt.imshow(mpimg.imread(originalimage+'.png'))
    print("The original image you have chosen is:",originalimage+'.png',"\nThis image will now be processed to 32x32.")
    plt.show(block=True)
    processedimage = imageprocessing.open(originalimage+'.png')
    processedimage = processedimage.resize((32, 32), imageprocessing.ANTIALIAS)
    processedimage.save(originalimage+'_database_processed.png') 
    plt.imshow(mpimg.imread(originalimage+'_database_processed.png'))
    print("The CIFAR-10 compatible image for our database is:",originalimage+'_database_processed.png',"\nThis image must be placed into the same folder as the 4.4 program.")

if __name__ == '__main__':
    main()