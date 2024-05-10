from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

model = load_model('C:/Dataset/RealWaste/Phanloairac_VGG16.h5') 

img_path = 'C:\Dataset\RealWaste\Miscellaneous Trash\Miscellaneous Trash_32.jpg'  

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  

predictions = model.predict(img_array)

predicted_label = np.argmax(predictions)

plt.imshow(img)
plt.axis('off')

if predicted_label == 0:
    plt.title('Đây là bìa các tông và thuộc rác tái chế')
elif predicted_label == 1:
    plt.title('Đây là thức ăn thừa và thuộc rác hữu cơ')
elif predicted_label == 2:
    plt.title('Đây là kính và thuộc rác tái chế')
elif predicted_label == 3:
    plt.title('Dây là kim loại và thuộc rác tái chế')
elif predicted_label == 4:
    plt.title('Đây là rác linh tinh và thuộc rác còn lại')
elif predicted_label == 5:
    plt.title('Đây là giấy và thuộc rác tái chế ')
elif predicted_label == 6:
    plt.title('Đây là nhựa và thuộc rác tái chế ')
elif predicted_label == 7:
    plt.title('Đây là vải và thuộc rác còn lại')
else:
    plt.title('Đây là rau củ quả và thuộc rác còn lại')

plt.show()  # Hiển thị hình ảnh và kết quả dự đoán
