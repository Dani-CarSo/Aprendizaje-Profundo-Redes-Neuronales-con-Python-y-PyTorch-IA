
from model import Net, get_dataloaders, train, evaluate, save_model, load_model # Importar funciones del módulo model.py
from visualize import explore_dataset, show_random_predictions, show_misclassified # Importar funciones del módulo visualize.py

def main(): # Función principal para ejecutar el flujo completo del proyecto
    # 1. Cargar datos
    print(" Cargando datos") # Mensaje para indicar que se están cargando los datos
    trainset, testset = get_dataloaders(batch_size=10) # Obtener los dataloaders para entrenamiento y prueba con un tamaño de batch de 10

    # 2. Explorar el dataset
    print("\nExploración del dataset ") # Mensaje para indicar que se va a explorar el dataset
    explore_dataset(trainset) # Llamar a la función para explorar el dataset de entrenamiento (visualización de imágenes y distribución de clases)

    # 3. Crear y entrenar la red
    print("\nEntrenando la red neuronal") # Mensaje para indicar que se va a entrenar la red neuronal
    net = Net() # Crear una instancia de la red neuronal definida en model.py
    print(net) # Imprimir la arquitectura de la red neuronal
    train(net, trainset, epochs=3, lr=0.001) # Entrenar la red neuronal durante 3 épocas con una tasa de aprendizaje de 0.001

    # 4. Evaluar
    print("\nEvaluación") # Mensaje para indicar que se va a evaluar el modelo
    evaluate(net, testset) # Evaluar el modelo en el conjunto de prueba y mostrar la precisión y matriz de confusión

    # 5. Guardar el modelo
    model_path = "mnist_nn_model.pth" # Ruta para guardar el modelo entrenado
    save_model(net, model_path) # Guardar el modelo entrenado en la ruta especificada

    # 6. Verificar carga del modelo
    print("\nVerificando carga del modelo") # Mensaje para indicar que se va a verificar la carga del modelo
    loaded_net = load_model(model_path) # Cargar el modelo desde la ruta especificada
    evaluate(loaded_net, testset) # Evaluar el modelo cargado para verificar que se ha guardado y cargado correctamente (debería tener la misma precisión que antes)

    # 7. Visualizaciones
    print("\n Predicciones aleatorias ") # Mensaje para indicar que se van a mostrar predicciones aleatorias
    show_random_predictions(loaded_net, testset, num_images=5) # Mostrar predicciones aleatorias del modelo cargado en el conjunto de prueba (5 imágenes)

    print("\nErrores de clasificación ") # Mensaje para indicar que se van a mostrar los errores de clasificación
    show_misclassified(loaded_net, testset, num_errors=5) # Mostrar imágenes mal clasificadas por el modelo cargado en el conjunto de prueba (5 errores)


if __name__ == "__main__": # Punto de entrada del programa
    main() # Llamar a la función principal para ejecutar el flujo completo del proyecto