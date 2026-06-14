
import torch # Importar la biblioteca PyTorch para trabajar con tensores y construir redes neuronales
import torch.nn as nn # Importar el módulo de redes neuronales de PyTorch para definir la arquitectura de la red
import torch.nn.functional as F # Importar el módulo de funciones de activación y pérdida de PyTorch para usar en la red
import torch.optim as optim # Importar el módulo de optimización de PyTorch para actualizar los pesos de la red durante el entrenamiento
from torchvision import transforms, datasets # Importar módulos de torchvision para transformar datos y cargar conjuntos de datos como MNIST


#  Arquitectura 
class Net(nn.Module): # Definir una clase Net que hereda de nn.Module, representando la arquitectura de la red neuronal
    """Red neuronal totalmente conectada de 4 capas para clasificar dígitos MNIST."""

    def __init__(self): # Método constructor para inicializar las capas de la red
        super().__init__() # Llamar al constructor de la clase base nn.Module para inicializar correctamente la red
        self.layer1 = nn.Linear(28 * 28, 64) # Primera capa totalmente conectada que toma una imagen de 28x28 píxeles (784 entradas) y la reduce a 64 neuronas
        self.layer2 = nn.Linear(64, 64) # Segunda capa totalmente conectada que toma las 64 neuronas de la capa anterior y las reduce a 64 neuronas
        self.layer3 = nn.Linear(64, 64) # Tercera capa totalmente conectada que toma las 64 neuronas de la capa anterior y las reduce a 64 neuronas
        self.layer4 = nn.Linear(64, 10) # Cuarta capa totalmente conectada que toma las 64 neuronas de la capa anterior y las reduce a 10 neuronas, representando las 10 clases de dígitos (0-9)

    def forward(self, x): # Método forward que define cómo se propagan los datos a través de la red
        x = F.relu(self.layer1(x)) # Aplicar la función de activación ReLU a la salida de la primera capa
        x = F.relu(self.layer2(x)) # Aplicar la función de activación ReLU a la salida de la segunda capa
        x = F.relu(self.layer3(x)) # Aplicar la función de activación ReLU a la salida de la tercera capa
        x = self.layer4(x) # La salida de la cuarta capa no se activa con ReLU porque se aplicará una función de pérdida que espera logits (valores sin activar)
        return F.log_softmax(x, dim=1) # Aplicar la función log_softmax a la salida de la cuarta capa para obtener probabilidades logarítmicas para cada clase, lo que es útil para la función de pérdida CrossEntropyLoss


#Datos 
def get_dataloaders(batch_size: int = 10): # Función para descargar el conjunto de datos MNIST y devolver los DataLoaders de entrenamiento y prueba
    """Descarga MNIST y devuelve los DataLoaders de entrenamiento y prueba."""
    transform = transforms.Compose([transforms.ToTensor()]) # Definir una transformación que convierte las imágenes a tensores, lo que es necesario para alimentar la red neuronal

    train = datasets.MNIST('', train=True,  download=True, transform=transform) # Descargar el conjunto de datos de entrenamiento de MNIST, aplicando la transformación definida
    test  = datasets.MNIST('', train=False, download=True, transform=transform) # Descargar el conjunto de datos de prueba de MNIST, aplicando la misma transformación

    trainset = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True) # Crear un DataLoader para el conjunto de entrenamiento que divide los datos en lotes del tamaño especificado y los mezcla aleatoriamente
    testset  = torch.utils.data.DataLoader(test,  batch_size=batch_size, shuffle=False) # Crear un DataLoader para el conjunto de prueba que divide los datos en lotes del tamaño especificado sin mezclarlos

    return trainset, testset # Devolver los DataLoaders de entrenamiento y prueba


# Entrenamiento 
def train(net: nn.Module, trainset, epochs: int = 3, lr: float = 0.001) -> list[float]: # Función para entrenar la red neuronal, que toma la instancia de la red, el DataLoader de entrenamiento, el número de épocas y la tasa de aprendizaje, y devuelve una lista con la pérdida del último lote de cada época
  
    loss_function = nn.CrossEntropyLoss()  # Definir la función de pérdida CrossEntropyLoss, que es adecuada para problemas de clasificación multiclase como MNIST
    optimizer = optim.Adam(net.parameters(), lr=lr) # Crear un optimizador Adam que actualizará los pesos de la red durante el entrenamiento, utilizando la tasa de aprendizaje especificada
    history = []  # Lista para almacenar la pérdida del último lote de cada época, lo que permitirá monitorear el progreso del entrenamiento

    net.train()  # Establecer la red en modo de entrenamiento, lo que activa comportamientos específicos como el dropout (si se usara) y permite que los gradientes se calculen durante la retropropagación
    for epoch in range(epochs): # Bucle principal de entrenamiento que se ejecuta durante el número de épocas especificado
        for data in trainset:  # Bucle que itera sobre los lotes de datos en el DataLoader de entrenamiento, donde cada lote contiene un conjunto de imágenes (X) y sus etiquetas correspondientes (y)
            X, y = data # Desempaquetar el lote de datos en las imágenes (X) y las etiquetas (y)
            net.zero_grad() # Limpiar los gradientes acumulados de la red antes de realizar la retropropagación, lo que es necesario para evitar que los gradientes se acumulen entre lotes
            output = net(X.view(-1, 784)) # Pasar las imágenes a través de la red para obtener las salidas (logits). Las imágenes se aplanan de 28x28 píxeles a un vector de 784 elementos para que coincidan con la entrada de la primera capa de la red
            loss = F.nll_loss(output, y)  # Calcular la pérdida utilizando la función de pérdida Negative Log Likelihood Loss, que es adecuada para las salidas logarítmicas producidas por log_softmax en la red
            loss.backward() # Realizar la retropropagación para calcular los gradientes de la pérdida con respecto a los pesos de la red
            optimizer.step() # Actualizar los pesos de la red utilizando el optimizador Adam, que ajusta los pesos en función de los gradientes calculados y la tasa de aprendizaje

        epoch_loss = loss.item()    # Obtener el valor numérico de la pérdida del último lote de la época actual y almacenarlo en la variable epoch_loss
        history.append(epoch_loss) # Agregar la pérdida del último lote de la época actual a la lista history para monitorear el progreso del entrenamiento a lo largo de las épocas
        print(f"Época {epoch + 1}/{epochs} — pérdida: {epoch_loss:.4f}") # Imprimir el número de época actual y la pérdida del último lote de esa época, formateada a cuatro decimales para facilitar la lectura

    return history # Devolver la lista con la pérdida del último lote de cada época, lo que permite analizar cómo evolucionó la pérdida durante el entrenamiento


# Evaluación
def evaluate(net: nn.Module, testset) -> float: # Función para evaluar la exactitud del modelo en el conjunto de prueba, que toma la instancia de la red y el DataLoader de prueba, y devuelve la exactitud como un valor entre 0.0 y 1.0
    """
    Evalúa la exactitud del modelo sobre el conjunto de prueba.

    Returns:
        Exactitud (0.0 – 1.0).
    """
    correct = 0 # Variable para contar el número de predicciones correctas realizadas por la red en el conjunto de prueba
    total = 0 # Variable para contar el número total de muestras evaluadas en el conjunto de prueba, lo que se utilizará para calcular la exactitud al final de la evaluación

    net.eval() # Establecer la red en modo de evaluación, lo que desactiva comportamientos específicos del entrenamiento como el dropout y asegura que los gradientes no se calculen durante la evaluación, lo que mejora el rendimiento y reduce el uso de memoria
    with torch.no_grad(): # Context manager que desactiva el cálculo de gradientes durante la evaluación, lo que mejora el rendimiento y reduce el uso de memoria, ya que no se necesitan los gradientes para la evaluación
        for X, y in testset: # Bucle que itera sobre los lotes de datos en el DataLoader de prueba, donde cada lote contiene un conjunto de imágenes (X) y sus etiquetas correspondientes (y)
            output = net(X.view(-1, 784)) # Pasar las imágenes a través de la red para obtener las salidas (logits). Las imágenes se aplanan de 28x28 píxeles a un vector de 784 elementos para que coincidan con la entrada de la primera capa de la red
            predictions = torch.argmax(output, dim=1)   # Obtener las predicciones de la red tomando el índice del valor máximo en la salida para cada muestra, lo que corresponde a la clase predicha por la red
            correct += (predictions == y).sum().item() # Comparar las predicciones con las etiquetas reales (y) y sumar el número de predicciones correctas al contador correct, utilizando .sum() para contar los aciertos en el lote y .item() para obtener el valor numérico
            total += len(y) # Sumar el número de muestras en el lote al contador total, utilizando len(y) para obtener el número de etiquetas en el lote, lo que se utilizará para calcular la exactitud al final de la evaluación

    accuracy = correct / total # Calcular la exactitud dividiendo el número de predicciones correctas por el número total de muestras evaluadas, lo que da un valor entre 0.0 y 1.0 que representa la proporción de predicciones correctas
    print(f"Exactitud en prueba: {accuracy:.3%}") # Imprimir la exactitud en el conjunto de prueba formateada como un porcentaje con tres decimales para facilitar la lectura
    return accuracy # Devolver la exactitud calculada, lo que permite utilizar este valor para comparar el rendimiento del modelo con otros modelos o para monitorear el progreso del entrenamiento a lo largo del tiempo

#  Guardar / cargar 

def save_model(net: nn.Module, path: str = "mnist_nn_model.pth") -> None: # Función para guardar los pesos del modelo en un archivo, que toma la instancia de la red y la ruta del archivo donde se guardarán los pesos
    torch.save(net.state_dict(), path) # Guardar los pesos de la red utilizando torch.save, que serializa el estado de la red (los pesos) en un archivo con la extensión .pth, lo que permite cargar estos pesos más tarde para usar el modelo sin necesidad de entrenarlo nuevamente
    print(f"Modelo guardado en: {path}") # Imprimir un mensaje indicando que el modelo ha sido guardado correctamente, mostrando la ruta del archivo donde se guardaron los pesos


def load_model(path: str = "mnist_nn_model.pth") -> Net: # Función para cargar los pesos del modelo desde un archivo, que toma la ruta del archivo donde se guardaron los pesos y devuelve una instancia de la red con esos pesos cargados
    net = Net() # Crear una nueva instancia de la clase Net, que representa la arquitectura de la red neuronal, pero sin pesos cargados aún
    net.load_state_dict(torch.load(path, weights_only=True)) # Cargar los pesos guardados en el archivo especificado utilizando torch.load para deserializar el estado de la red y load_state_dict para cargar esos pesos en la instancia de la red creada, lo que permite restaurar el modelo a su estado entrenado previamente
    net.eval() # Establecer la red en modo de evaluación después de cargar los pesos, lo que desactiva comportamientos específicos del entrenamiento como el dropout y asegura que los gradientes no se calculen durante la evaluación, lo que mejora el rendimiento y reduce el uso de memoria al usar el modelo para hacer predicciones
    print(f"Modelo cargado desde: {path}") # Imprimir un mensaje indicando que el modelo ha sido cargado correctamente, mostrando la ruta del archivo desde donde se cargaron los pesos
    return net # Devolver la instancia de la red con los pesos cargados, lo que permite usar este modelo para hacer predicciones o para continuar entrenándolo si se desea

# Script principal 
if __name__ == "__main__":  # Verificar si el script se está ejecutando directamente (en lugar de ser importado como un módulo), lo que permite ejecutar el código de entrenamiento y evaluación solo cuando se ejecuta este archivo
    trainset, testset = get_dataloaders() # Llamar a la función get_dataloaders para obtener los DataLoaders de entrenamiento y prueba, lo que prepara los datos para ser usados en el entrenamiento y evaluación del modelo

    net = Net() # Crear una instancia de la clase Net, que representa la arquitectura de la red neuronal, lo que permite usar esta instancia para entrenar el modelo y hacer predicciones
    print(net) # Imprimir la arquitectura de la red neuronal, lo que muestra las capas y el número de parámetros en cada capa, lo que puede ser útil para entender la complejidad del modelo

    train(net, trainset, epochs=3)  # Llamar a la función train para entrenar la red neuronal utilizando el DataLoader de entrenamiento, especificando el número de épocas y la tasa de aprendizaje, lo que ajusta los pesos de la red para mejorar su rendimiento en la tarea de clasificación de dígitos MNIST
    evaluate(net, testset)  # Llamar a la función evaluate para evaluar la exactitud del modelo en el conjunto de prueba utilizando el DataLoader de prueba, lo que permite medir el rendimiento del modelo en datos que no se usaron durante el entrenamiento
    save_model(net) # Llamar a la función save_model para guardar los pesos del modelo entrenado en un archivo, lo que permite conservar el modelo para su uso futuro sin necesidad de entrenarlo nuevamente