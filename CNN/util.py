import numpy as np

# Unscramble the shuffled jigsaw image using inputed label
def rearrange(label, fourPiece_img):
    index_topleft = np.where(label == 0)[0][0]
    index_topright = np.where(label == 1)[0][0]
    index_bottleft = np.where(label == 2)[0][0]
    index_bottright = np.where(label == 3)[0][0]
    
    output_img = np.zeros((200,200,3))
    output_img[:100,:100,:] = fourPiece_img[index_topleft]
    output_img[:100,100:,:] = fourPiece_img[index_topright]
    output_img[100:,:100,:] = fourPiece_img[index_bottleft]
    output_img[100:,100:,:] = fourPiece_img[index_bottright]
    
    return output_img