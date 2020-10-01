from keras.layers import Conv2D, Reshape,Concatenate,Dense, Lambda, Conv1D,UpSampling2D,LeakyReLU,BatchNormalization,Input,Flatten
from keras.models import Model

def generate_block(image_in):
    image_in = Conv2D(256,3,padding="same")(image_in)
    image_in = LeakyReLU()(image_in)

    image_in = Conv2D(256,3,strides=2,padding="same")(image_in)
    image_in = LeakyReLU()(image_in)

    return image_in
def generate_model(inp,upsizes = 4):
    for i in range(upsizes-1):
        inp = generate_block(inp)
    b1 = Flatten()(inp)
    b1 = Dense(256)(b1)
    b1 = Dense(1)(b1)
    return b1
def Discriminator(upsizes = 4):
    inp = Input((4*2**upsizes,4*2**upsizes,3))
    out =generate_model(inp,upsizes)

    m = Model(inp,out)
    return m
