import os
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Tạo generators cho dữ liệu huấn luyện và kiểm tra
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Tải dữ liệu và tạo generators
train_generator = train_datagen.flow_from_directory(
    'C:\Dataset\RealWaste',  # Đường dẫn tới thư mục dữ liệu huấn luyện
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    'C:\Dataset\waste test',  # Đường dẫn tới thư mục dữ liệu kiểm tra
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load pre-trained VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Tạo mô hình mới và thêm các layers
model = Sequential()
model.add(vgg_model)
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(9, activation='softmax'))

# Compile mô hình
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(train_generator, epochs=30, validation_data=test_generator)

# Đánh giá độ chính xác của mô hình trên dữ liệu kiểm tra
score = model.evaluate(test_generator, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Lưu mô hình
model.save('Phanloairac_VGG16.h5')
