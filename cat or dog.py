import os
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pylab as plt

#创建一个模型并设置参数
pre_trained_model = InceptionV3(input_shape=(150,150,3),
                                include_top=False)


#模型每一场训练不进行调参数
for layer in pre_trained_model.layers:
    layer.trainable = False


pre_trained_model.summary()

#取出对应的层数
last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

#最后一层进行展平
x = layers.Flatten()(last_output)
#定义全连接层，输出全连接层
x = layers.Dense(1024,activation='relu')(x)
x = layers.Dense(1,activation='sigmoid')(x)

#构建自己的模型
model = Model(pre_trained_model.input,x)
model.compile(optimizer=RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['acc'])

#数据集位置路径读取
base_dir = 'cats_and_dogs_filtered'
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')

#数据集处理
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
test_datagen =ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    batch_size=20,
    class_mode='binary',
    target_size=(150,150)
)
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    batch_size=20,
    class_mode='binary',
    target_size=(150,150)
)

#训练参数的设置
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=100,
    epochs=20,
    validation_steps=50,
    verbose=2
)

#取出数据
#训练集准确率
acc = history.history['acc']
#验证集准确率
val_acc = history.history['val_acc']
#损失值
loss = history.history['loss']
val_loss = history.history['val_loss']

#可视化操作
epochs = range(len(acc))
plt.plot(epochs,acc)
plt.plot(epochs,val_loss)
plt.title('training and validation accuracy')
