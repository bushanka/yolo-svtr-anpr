import onnxruntime

def get_onnx_model_summary(model_path):
    # Создание сессии ONNX Runtime
    sess = onnxruntime.InferenceSession(model_path)

    # Получение информации о входных и выходных узлах модели
    input_info = sess.get_inputs()
    output_info = sess.get_outputs()

    # Вывод сводки модели
    print("Сводка модели:")
    print("Входные узлы:")
    for i, input in enumerate(input_info):
        print(f"  {i+1}. Имя: {input.name}, тип: {input.type}, форма: {input.shape}")

    print("Выходные узлы:")
    for i, output in enumerate(output_info):
        print(f"  {i+1}. Имя: {output.name}, тип: {output.type}, форма: {output.shape}")

# Укажите путь к модели ONNX
model_path = "/home/bush/project/anpr/anpr/models/detection/weights/yolo_int8_sim.onnx"

# Вызов функции для получения сводки модели
get_onnx_model_summary(model_path)