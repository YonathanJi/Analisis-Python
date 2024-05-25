#!/usr/bin/env python
# coding: utf-8

# Autor: Luis Carlos Rodríguez Timaná
# 
# correo: luistimana.00@gmail.com

# # Regresión Lineal Simple
# 
# La regresión lineal es un método estadístico que se utiliza para modelar la relación entre una variable dependiente y una o más variables independientes mediante una línea recta. El objetivo es encontrar la ecuación de esa línea, que puede ser utilizada para hacer predicciones sobre la variable dependiente basándose en los valores de las variables independientes.
# 
# _y_ = _mx_ + _b_
# 
# El objetivo en la regresión lineal es encontrar los valores _m_ y _b_ que minimizan la suma de los cuadrados de los residuos, es decir, la diferencia entre los valores observados y los valores predichos por el modelo.

# <img src="./1.jpeg">

# # Ejemplo de regresión lineal simple

# In[1]:


import matplotlib.pyplot as plt
import numpy as np

horasEstudio = [1, 2, 3, 4, 6 ,8]
nota = [2.2, 2.8, 3.4, 3.9, 4.2, 4.8]

# Graficar en un plano cartesiano
plt.scatter(horasEstudio, nota, color='red')

# Configurar etiquetas y título
plt.xlabel('Horas Estudio')
plt.ylabel('Nota')
plt.title('Horas de estudio VS Nota')


# # Predicción del precio de las casas con regresión Lineal Simple
# 
# Descripción de la base de datos:
# 
# - Id: Identificador único para cada observación o vivienda.
# - MSSubClass: Clasificación de la construcción.
# - MSZoning: Zonificación del área.
# - LotFrontage: Longitud de frente del lote.
# - LotArea: Área total del lote.
# - Street: Tipo de acceso a la propiedad (pavimentado o no pavimentado).
# - Alley: Tipo de calle de acceso al callejón.
# - LotShape: Forma del lote.
# - LandContour: Contorno de la tierra.
# - Utilities: Servicios públicos disponibles.
# - Neighborhood: Vecindario en el que se encuentra la propiedad.
# - Condition1, Condition2: Condiciones generales de proximidad a diversas características (carretera, etc.).
# - BldgType: Tipo de vivienda (unifamiliar, dúplex, etc.).
# - HouseStyle: Estilo de la vivienda.
# - OverallQual: Calidad general de la casa.
# - OverallCond: Condición general de la casa.
# - YearBuilt: Año de construcción de la vivienda.
# - YearRemodAdd: Año de la última remodelación.
# - RoofStyle: Estilo del techo.
# - RoofMatl: Material del techo.
# - Exterior1st, Exterior2nd: Revestimiento exterior de la casa.
# - MasVnrType: Tipo de revestimiento de mampostería.
# - MasVnrArea: Área de revestimiento de mampostería en pies cuadrados.
# - ExterQual: Calidad del material exterior.
# - ExterCond: Condición del material exterior.
# - Foundation: Tipo de cimentación.
# - BsmtQual: Calidad del sótano.
# - BsmtCond: Condición del sótano.
# - BsmtExposure: Exposición del sótano.
# - BsmtFinType1: Calidad de terminación del sótano.
# - BsmtFinSF1: Pies cuadrados terminados del área del sótano tipo 1.
# - BsmtFinType2: Calidad de terminación del sótano (si hay más de un tipo).
# - BsmtFinSF2: Pies cuadrados terminados del área del sótano tipo 2.
# - BsmtUnfSF: Pies cuadrados sin terminar del área del sótano.
# - TotalBsmtSF: Área total del sótano en pies cuadrados.
# - Heating: Tipo de sistema de calefacción.
# - HeatingQC: Calidad y condición del sistema de calefacción.
# - CentralAir: Aire acondicionado central.
# - Electrical: Sistema eléctrico.
# - 1stFlrSF, 2ndFlrSF: Pies cuadrados del primer y segundo piso, respectivamente.
# - LowQualFinSF: Pies cuadrados terminados de baja calidad (todos los pisos).
# - GrLivArea: Área habitable sobre el nivel del suelo.
# - BsmtFullBath, BsmtHalfBath: Número de baños completos y medios en el sótano, respectivamente.
# - FullBath, HalfBath: Número de baños completos y medios en la casa, respectivamente.
# - BedroomAbvGr: Número de dormitorios por encima del nivel del sótano.
# - KitchenAbvGr: Número de cocinas.
# - KitchenQual: Calidad de la cocina.
# - TotRmsAbvGrd: Total de habitaciones por encima del nivel del sótano (sin contar baños).
# - Functional: Funcionalidad de la casa.
# - Fireplaces: Número de chimeneas.
# - FireplaceQu: Calidad de la chimenea.
# - GarageType: Ubicación del garaje.
# - GarageYrBlt: Año de construcción del garaje.
# - GarageFinish: Acabado interior del garaje.
# - GarageCars: Capacidad de autos en el garaje.
# - GarageArea: Área del garaje en pies cuadrados.
# - GarageQual, GarageCond: Calidad y condición del garaje, respectivamente.
# - PavedDrive: Tipo de acceso pavimentado al garaje.
# - WoodDeckSF, OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch: Áreas de porches y terrazas.
# - PoolArea: Área de la piscina en pies cuadrados.
# - PoolQC: Calidad de la piscina.
# - Fence: Calidad de la cerca.
# - MiscFeature: Otras características misceláneas no cubiertas en otras categorías.
# - MiscVal: Valor de características misceláneas.
# - MoSold: Mes de venta.
# - YrSold: Año de venta.
# - SaleType: Tipo de venta.
# - SaleCondition: Condición de venta.
# - SalePrice: Precio de venta de la vivienda (variable objetivo).

# In[2]:


# lectura de datos en Python
import pandas as pd

train = pd.read_csv('./train.csv')


# In[3]:


train.head()


# In[4]:


train.columns


# In[5]:


train[['GrLivArea','SalePrice']].head(10)  #Superficie habitable por encima del nivel del suelo en pies cuadrados
                                           #Precio de venta


# In[6]:


train.plot.scatter(x='GrLivArea',y='SalePrice')
plt.show()


# In[7]:


# pintando una línea recta sobre los datos
# y = wx + b


# In[8]:


# parametros de la recta
w = 118 #m
b = 0

# y = wx + b


# In[9]:


# puntos de la recta
x = np.linspace(0,train['GrLivArea'].max())
y = w*x+b

# grafica de la recta
train.plot.scatter(x='GrLivArea',y='SalePrice',s=10) #nombre de los ejes
plt.plot(x, y, '-r')
plt.ylim(0,train['SalePrice'].max()*1.2) #ajuste del eje y
plt.grid()
plt.show()


# In[10]:


# si escogemos esos parametros para el modelo, ¿cual es el error?


# In[11]:


# calculo de las predicciones
train['pred'] = train['GrLivArea']*w + b   #y=wx+b
train.head()


# In[12]:


train[['GrLivArea','SalePrice','pred']].head()


# In[13]:


# calculo de la funcion de error
train['diff'] = train['pred']-train['SalePrice'] #diferencia de las predicciones y valor real del eje y
train['cuad'] = train['diff']**2
train


# In[14]:


train[['GrLivArea','SalePrice','pred', 'diff', 'cuad']].head()


# In[15]:


#Error cuadrático medio
train['cuad'].mean()/1000000


# # Simplificar el proceso

# In[16]:


# grid de la funcion de error basado en m, b=0
# q = np.linspace(1,100,10);
w = np.linspace(50,200,150)
grid_error = pd.DataFrame(w, columns=['w'])
grid_error;


# In[17]:


def sum_error(w, train):
    b=0
    train['pred'] = train['GrLivArea']*w+b
    train['diff'] = train['pred']-train['SalePrice']
    train['cuad'] = train['diff']**2
    return(train['cuad'].mean())
train['cuad'].mean()


# In[18]:


# a cada w se le asigna su error cuadrático medio. Es como si se ejecutara lo de arriba pero para distintas w.
grid_error['error'] = grid_error['w'].apply(lambda x: sum_error(x, train))
#grid_error['error'].min()
#grid_error['error']
#grid_error['w']
     
#Nuevo DF para extraer el w mínimo.
grid_error_menor = pd.DataFrame()
grid_error_menor['menorW'] = [ grid_error['w'] [grid_error['error'].argmin()]]
errorMinimo = float(grid_error_menor['menorW'])
errorMinimo


# # Interpretación
# 
# Por cada aumento de un pié cuadrado, aumenta 118 dolares el precio de las casas.

# In[19]:


grid_error.plot(x='w',y='error')
plt.show()


# # Ejercicio en clase
# 
# - Buscar otra variable de la base de datos, que permita determinar una relación con la variable dependiente (SalePrice).
# - Determinar los parámetros subyacentes (la intersección y la pendiente) de la ecuación de la recta que mejor se ajuste a los datos.

# # Implementando la librería Sklearn

# In[20]:


import numpy as np

# Crear una matriz unidimensional
array_unidimensional = np.array([1, 2, 3, 4, 5, 6])

# Utilizar reshape para convertir la matriz unidimensional en una matriz bidimensional (3 filas x 2 columnas)
array_bidimensional = array_unidimensional.reshape((3, 2))

# Imprimir las matrices originales y la transformada
print("Matriz Unidimensional:")
print(array_unidimensional)
print("\nMatriz Bidimensional:")
print(array_bidimensional)


# In[21]:


import numpy as np

# Crear una matriz unidimensional
array_unidimensional = np.array([1, 2, 3, 4, 5, 6])

# Utilizar reshape para convertir la matriz unidimensional en una matriz bidimensional con una columna
array_bidimensional = array_unidimensional.reshape((-1, 1))

# Imprimir las matrices originales y la transformada
print("Matriz Unidimensional:")
print(array_unidimensional)
print("\nMatriz Bidimensional:")
print(array_bidimensional)


# In[22]:


# usando sklear para saber los valores optimos
from sklearn.linear_model import LinearRegression

# definiendo input y output

X_train = np.array(train['GrLivArea']).reshape((-1, 1)) #Primer variable (entrada)
Y_train = np.array(train['SalePrice'])                  #Segunda variable (salida)

# creando modelo
model = LinearRegression(fit_intercept=True) #la línea de regresión pasa por el origen (0,0).
model.fit(X_train, Y_train)

# imprimiendo parametros
print(f"intercepto (b): {model.intercept_}")
print(f"pendiente (w): {model.coef_}")


# In[23]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(train[['GrLivArea']], train['SalePrice'], test_size=0.2, random_state=42)

# Crear y ajustar el modelo de regresión lineal con scikit-learn
model = LinearRegression(fit_intercept=False)
model.fit(X_train, Y_train)

# Realizar predicciones en el conjunto de prueba
predictions_test = model.predict(X_test)

# Calcular el Error Cuadrático Medio (MSE) en el conjunto de prueba
mse_test = mean_squared_error(Y_test, predictions_test)
print(f"Error Cuadrático Medio (MSE) en prueba: {mse_test}")

# Añadir una columna de unos para el término constante
X_train_sm = sm.add_constant(X_train)

# Crear y ajustar el modelo de regresión lineal con statsmodels
model_sm = sm.OLS(Y_train, X_train_sm).fit()

# Imprimir un resumen completo del modelo con statsmodels
print(model_sm.summary())

# Visualizar la regresión en el conjunto de entrenamiento
plt.scatter(X_train, Y_train, label='Datos de entrenamiento')
plt.plot(X_train, model_sm.predict(sm.add_constant(X_train)), color='red', label='Regresión lineal')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.title('Regresión Lineal - Datos de Entrenamiento')
plt.legend()
plt.show()

# Del resultado:
# const = intercepto
# GrLivArea = pendiente

# R squared (coheficiente de determinación) < 0.5 mal ajuste, >= 0.5 y <0.7 aceptable, >=0.7 y <0.9 buen ajuste, >0.9 y <1 es muy buen ajuste.


# # Método de los Mínimos Cuadrados
# 
# Los mínimos cuadrados se utilizan para ajustar un modelo a datos observados y encontrar los parámetros del modelo que mejor se ajustan a esos datos. Este enfoque es ampliamente utilizado en estadísticas y aprendizaje automático para entrenar modelos y hacer predicciones.

# <img src="./2.jpeg" width="400">

# <img src="./3.jpeg" width="300">

# In[24]:


# Aplicando mínimos cuadrados

# Calcular las sumas necesarias
n = len(train)
sum_x = train['GrLivArea'].sum()
sum_y = train['SalePrice'].sum()
sum_xy = (train['GrLivArea'] * train['SalePrice']).sum()
sum_x_squared = (train['GrLivArea'] ** 2).sum()


# In[25]:


# Calcular la pendiente (m) y la ordenada al origen (b)
m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
b = (sum_y - m * sum_x) / n

# Imprimir los resultados
print(f"Pendiente (m): {m}")
print(f"Ordenada al origen (b): {b}")


# In[26]:


regression_line = lambda x: m * x + b

# Visualizar los datos y la recta de regresión
plt.scatter(train['GrLivArea'], train['SalePrice'], label='Datos')
plt.plot(train['GrLivArea'], regression_line(train['GrLivArea']), color='red', label='Recta de Regresión')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.legend()
plt.title('Regresión Lineal')
plt.show()


# In[27]:


plt.plot(train['GrLivArea'], regression_line(train['GrLivArea']), color='red', label='Recta de Regresión')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.legend()
plt.title('Curva del modelo de predicción')
plt.show()


# In[28]:


# Realizando una predicción con la ecuación

x = 2000

prediccion = 107*x + 18570
print(f"El costo para una vivienda de {x} pies cuadrados es de {prediccion} dolares")


# In[29]:


x = int(input("Ingresa el tamaño de la vivienda: "))

prediccion = 107*x + 18570
print(f"El costo para una vivienda de {x} pies cuadrados ({x/10.764} metros cuadrados) es de {prediccion} dolares")


# # Ejercicio en clase
# 
# - Para la otra variable encontrada, aplicar los algoritmos de regresión lineal simple con sklearn.
# - Comparar la pendiente y el intercepto, con el método de mínimos cuadrados.

# Aplicar el análisis de regresión lineal simple (encontrando m y b con el algoritmo , aplicando el método de mínimos cuadrados y utilizando la librería sklearn) a la base de datos de "Wine Quality". Iniciar con un análisis de columnas simple y una descripción de la base de datos (visto en la sección anterior de pandas).
# 
# Descripción de la base de datos:
# 
# - Acidez Fija: representa la cantidad de ácido fijo presente en el vino. Los ácidos fijos son aquellos que no se evaporan fácilmente.
# 
# - Acidez Volátil: mide la cantidad de ácidos volátiles en el vino. Los ácidos volátiles pueden contribuir a olores desagradables si están presentes en cantidades elevadas.
# 
# - Ácido Cítrico: indica la cantidad de ácido cítrico en el vino, que puede afectar la frescura y acidez.
# 
# - Azúcar Residual: representa la cantidad de azúcares que quedan después de la fermentación. Afecta la dulzura del vino.
# 
# - Cloruros: mide la cantidad de cloruros presentes en el vino, los cuales pueden influir en el sabor.
# 
# - Dióxido Libre de Azufre: indica la cantidad de dióxido de azufre no unido a otras sustancias. El azufre es un conservante utilizado en el vino.
# 
# - Total Dióxido de Azufre: representa la suma de dióxido de azufre libre y dióxido de azufre unido. Contribuye a la estabilidad y conservación del vino.
# 
# - Densidad: es la densidad del vino, que puede estar relacionada con la cantidad de azúcar y alcohol.
# 
# - pH: indica el nivel de acidez o alcalinidad del vino. Un pH bajo indica acidez, mientras que un pH alto indica alcalinidad.
# 
# - Sulfatos: mide la cantidad de sulfatos presentes en el vino. Pueden contribuir a la preservación del vino y prevenir la oxidación.
# - Alcohol: representa el contenido de alcohol en el vino. Es un factor importante que influye en el sabor y cuerpo del vino.
# 
# - Calidad: es la variable de respuesta. Representa la calidad del vino.

# In[30]:


vino = pd.read_csv("Wine Quality.txt", sep=";" )
vino.head()


# In[31]:


#Solo se puede cuando cambio la misma cantidad de de nombres
nombres = ["acidez fija", "acidez volátil", "ácido cítrico", "azúcar residual", "cloruros", "dióxido libre de azufre", "total dióxido de azufre", "densidad", "pH", "sulfatos", "alcohol", "calidad"]
vino.columns = nombres
vino.head()


# In[32]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[33]:


df = pd.DataFrame(vino)

# Calcular la matriz de correlación
correlation_matrix = df.corr()

# Crear un mapa de correlación con un tamaño más grande
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Mapa de Correlación')
plt.show()


# In[34]:


# Aplicando mínimos cuadrados

# Calcular las sumas necesarias
n = len(vino)
sum_x = vino['acidez fija'].sum()
sum_y = vino['pH'].sum()
sum_xy = (vino['acidez fija'] * vino['pH']).sum()
sum_x_squared = (vino['acidez fija'] ** 2).sum()


# In[35]:


# Calcular la pendiente (m) y la ordenada al origen (b)
m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
b = (sum_y - m * sum_x) / n

# Imprimir los resultados
print(f"Pendiente (m): {m}")
print(f"Ordenada al origen (b): {b}")


# In[36]:


regression_line = lambda x: m * x + b

# Visualizar los datos y la recta de regresión
plt.scatter(vino['acidez fija'], vino['pH'], label='Datos')
plt.plot(vino['acidez fija'], regression_line(vino['acidez fija']), color='red', label='Recta de Regresión')
plt.xlabel('acidez fija')
plt.ylabel('pH')
plt.legend()
plt.title('Regresión Lineal')
plt.show()


# In[37]:


m = -0.06056103155684543
b = 3.8149590111969167
x = 200

y = m*x + b

y



# In[38]:


# usando sklear para saber los valores optimos
from sklearn.linear_model import LinearRegression

# definiendo input y output

X_vino = np.array(vino['acidez fija']).reshape((-1, 1)) #Primer variable (entrada)
Y_vino = np.array(vino['pH'])                  #Segunda variable (salida)

# creando modelo
model = LinearRegression(fit_intercept=True) #la línea de regresión pasa por el origen (0,0).
model.fit(X_vino, Y_vino)

# imprimiendo parametros
print(f"intercepto (b): {model.intercept_}")
print(f"pendiente (w): {model.coef_}")


# In[40]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# In[43]:


# Dividir los datos en conjuntos de entrenamiento y prueba
X_vino, X_test, Y_vino, Y_test = train_test_split(vino[['acidez fija']], vino['pH'], test_size=0.2, random_state=42)

# Crear y ajustar el modelo de regresión lineal con scikit-learn
model = LinearRegression(fit_intercept=False)
model.fit(X_vino, Y_vino)

# Realizar predicciones en el conjunto de prueba
predictions_test = model.predict(X_test)

# Calcular el Error Cuadrático Medio (MSE) en el conjunto de prueba
mse_test = mean_squared_error(Y_test, predictions_test)
print(f"Error Cuadrático Medio (MSE) en prueba: {mse_test}")

# Añadir una columna de unos para el término constante
X_vino_sm = sm.add_constant(X_vino)

# Crear y ajustar el modelo de regresión lineal con statsmodels
model_sm = sm.OLS(Y_vino, X_vino_sm).fit()

# Imprimir un resumen completo del modelo con statsmodels
print(model_sm.summary())

# Visualizar la regresión en el conjunto de entrenamiento
plt.scatter(X_vino, Y_vino, label='Datos de entrenamiento')
plt.plot(X_vino, model_sm.predict(sm.add_constant(X_vino)), color='red', label='Regresión lineal')
plt.xlabel('acidez fija')
plt.ylabel('pH')
plt.title('Regresión Lineal - Datos de Entrenamiento')
plt.legend()
plt.show()

# Del resultado:
# const = intercepto
# GrLivArea = pendiente

# R squared (coheficiente de determinación) < 0.5 mal ajuste, >= 0.5 y <0.7 aceptable, >=0.7 y <0.9 buen ajuste, >0.9 y <1 es muy buen ajuste.

