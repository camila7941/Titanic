0# PROYECTO FINAL - LIMPIEZA DEL DATASET TITANIC

from itertools import combinations
import re
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_seq_items", None)

# True muestra todas las consultas de análisis false deja el script en silencio.
DIAGNOSTICO = False

# CONSTANTES
# Tarifa más alta registrada en el Titanic original
FARE_MAX = 512.3292

# Tope de tarifa por clase
TOPE_POR_CLASE = {1: 512.3292, 2: 75.0, 3: 70.0}

# Formas estándar de los prefijos de Ticket.
PREFIJOS_ESTANDAR = {
    "A4":       "A/4",
    "A5":       "A/5",
    "AQ3":      "AQ/3",
    "AQ4":      "AQ/4",
    "C":        "C",
    "CA":       "C.A.",
    "CASOTON":  "C.A./SOTON",
    "FA":       "FA",
    "FC":       "F.C.",
    "FCC":      "F.C.C.",
    "LINE":     "LINE",
    "PC":       "PC",
    "PP":       "P.P.",
    "PPP":      "P/PP",
    "SC":       "SC",
    "SCA3":     "SC/A.3",
    "SCA4":     "SC/A4",
    "SCAH":     "SC/AH",
    "SCOW":     "SCO/W",
    "SCPARIS":  "SC/PARIS",
    "SOC":      "S.O.C.",
    "SOP":      "S.O.P.",
    "SOPP":     "S.O./P.P.",
    "SOTONO2":  "SOTON/O2",
    "SOTONOQ":  "SOTON/O.Q.",
    "STONO2":   "STON/O2.",
    "SWPP":     "SW/PP",
    "WC":       "W./C.",
    "WEP":      "WE/P",
}

# FUNCIONES 

def distancia(a, b):
    
    prev = np.arange(len(b) + 1)
    for i, ca in enumerate(a, 1):
        cur = np.zeros(len(b) + 1, dtype=int)
        cur[0] = i
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def limpiar_key(s):
    if pd.isna(s):
        return s
    palabras = [p for p in s.split() if len(p) > 2 or p in ("jr",)]
    return " ".join(palabras)


# CARGAR EL ARCHIVO
df = pd.read_csv("Titanic_sucio.csv", sep=";", encoding="latin1")

filas_iniciales = len(df)


# 1. DIAGNOSTICO
if DIAGNOSTICO:
    print(df.shape)
    print("Filas idénticas:", df.duplicated().sum())
    print("PassengerId duplicados:", df["PassengerId"].duplicated().sum())

    for col in ["Sex", "Embarked", "Pclass", "Survived"]:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False))

    print("\nNulos por columna:")
    print(df.isna().sum())
    print("\nTipos de datos:")
    print(df.dtypes)


# 2. NORMALIZAR Name Y EXTRAER Titulo / Apellido
df["Name"] = (
    df["Name"]
    .str.replace('"', "", regex=False)     
    .str.replace(r"\s+", " ", regex=True)  
    .str.strip()
)

# La regex se ancla al punto del título, no a la coma
df["Titulo"] = df["Name"].str.extract(r"\b([A-Za-z]+)\.", expand=False)
df["Apellido"] = df["Name"].str.extract(r"^(.+?)\s+[A-Za-z]+\.", expand=False)

# NORMALIZAR NOMBRE: sin título, mayúsculas, tildes ni símbolos. 
df["name_key"] = (
    df["Name"]
    .str.replace(r"\b[A-Za-z]+\.", "", regex=True)
    .str.lower()
    .str.normalize("NFKD")
    .str.encode("ascii", "ignore")
    .str.decode("utf-8")
    .str.replace(r"[^a-z ]", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)


# 3. CORREGIR PassengerId

pid = df["PassengerId"].astype(str)
pid = (
    pid.str.replace("O", "0", regex=False)
    .str.replace("o", "0", regex=False)
    .str.replace("l", "1", regex=False)
    .str.replace("I", "1", regex=False)
    .str.replace("B", "8", regex=False)
    .str.replace("S", "5", regex=False)
)
pid = pid.str.replace(r"[^0-9]", "", regex=True)
df["PassengerId"] = pd.to_numeric(pid, errors="coerce")

validos = df[df["PassengerId"].between(1, 891)]
mapa_id = validos.groupby("name_key")["PassengerId"].first()
fuera_rango = df["PassengerId"] > 891
df.loc[fuera_rango, "PassengerId"] = df.loc[fuera_rango, "name_key"].map(mapa_id)


# 4. RECUPERAR IDs POR NOMBRE
# se busca el nombre más parecido con distancia de edición.

df["name_key2"] = df["name_key"].apply(limpiar_key)

validos = df[df["PassengerId"].between(1, 891)]
mapa_id = validos.groupby("name_key2")["PassengerId"].first()
candidatos = mapa_id.index.tolist()

sin_id = df["PassengerId"].isna() & df["name_key2"].notna()
recuperados = 0

for idx in df[sin_id].index:
    nombre = df.at[idx, "name_key2"]
    d_min, mejor = min((distancia(nombre, c), c) for c in candidatos)

    # Sin este if, min() siempre devuelve un valor,
    # aunque sea una persona completamente distinta.
    if d_min <= 6:
        df.at[idx, "PassengerId"] = mapa_id[mejor]
        recuperados += 1
        if DIAGNOSTICO:
            print(f"d={d_min} | {nombre}  ->  {mejor}")


# 5. CLASIFICAR IDs REPETITIVOS
# misma persona, fila copiada
# personas diferentes pero con el mismo ID
UMBRAL_COLISION = 10
conteo_id = df["PassengerId"].value_counts()
repetidos = conteo_id[conteo_id > 1].index
clasificacion = {}

for pid_valor in repetidos:
    nombres = df.loc[df["PassengerId"] == pid_valor, "name_key2"].dropna().unique()

    if len(nombres) == 0:
        clasificacion[pid_valor] = "sin_nombre"
    elif len(nombres) == 1:
        clasificacion[pid_valor] = "duplicado"
    else:
        d = max(distancia(a, b) for a, b in combinations(nombres, 2))
        clasificacion[pid_valor] = "colision" if d > UMBRAL_COLISION else "duplicado"

df["caso"] = df["PassengerId"].map(clasificacion).fillna("unico")

if DIAGNOSTICO:
    print(df["caso"].value_counts())
    print("\nColisiones:")
    print(
        df[df["caso"] == "colision"][["PassengerId", "Name", "Sex", "Age"]]
        .sort_values("PassengerId")
        .to_string()
    )


# 6. DESCARTAR DUPLICADOR Y DEJAR FILA MAS COMPLETA
# n_nulos: cuántos campos vacíos tiene cada fila.
# axis=1 suma horizontalmente (por fila, no por columna).
df["n_nulos"] = df.isna().sum(axis=1)

sin_id = df[df["PassengerId"].isna()]
colisiones = df[df["PassengerId"].notna() & (df["caso"] == "colision")]
normales = df[df["PassengerId"].notna() & (df["caso"] != "colision")]

# De cada grupo se conserva la fila más completa
dedup = normales.sort_values("n_nulos").drop_duplicates("PassengerId", keep="first")

df = pd.concat([dedup, colisiones, sin_id]).drop(columns="n_nulos")
df = df.sort_values("PassengerId").reset_index(drop=True)


# Copias con nombre idéntico bajo distinto ID:
# se conserva la del ID menor, que es el original.
con_nombre = df[df["name_key2"].notna()]
sin_nombre = df[df["name_key2"].isna()]
con_nombre = con_nombre.sort_values("PassengerId").drop_duplicates(
    "name_key2", keep="first"
)

df = pd.concat([con_nombre, sin_nombre])
df = df.sort_values("PassengerId").reset_index(drop=True)



# la llave es la combinación PassengerId + Apellido, para que dos personas distintas con el mismo ID no se pisen.
df["n_nulos"] = df.isna().sum(axis=1)
completas = df[df["PassengerId"].notna() & df["Apellido"].notna()]
incompletas = df[df["PassengerId"].isna() | df["Apellido"].isna()]

completas = completas.sort_values(["PassengerId", "n_nulos"]).drop_duplicates(
    ["PassengerId", "Apellido"], keep="first"
)

df = pd.concat([completas, incompletas]).drop(columns="n_nulos")
df = df.sort_values("PassengerId").reset_index(drop=True)
df = df.drop(columns=["name_key", "name_key2", "caso"], errors="ignore")

# FUNCIONES DE LIMPIEZA POR COLUMNA

""" agrupa los valores raros con el valor frecuente más parecido."""
def limpiar(df, col, umbral=10, dist_max=4):
    df[col] = df[col].astype(str).str.strip().str.capitalize()
    tabla = df[col].value_counts().reset_index()
    tabla.columns = [col, "conteo"]

    frecuentes = tabla.loc[tabla["conteo"] >= umbral, col].to_numpy()
    raros = tabla.loc[tabla["conteo"] < umbral, col].to_numpy()

    # Guardar valores para comparar
    if len(frecuentes) == 0:
        return df

    correcciones = {}
    for v in raros:
        d = np.array([distancia(v, f) for f in frecuentes])
        if d.size > 0 and d.min() <= dist_max:
            correcciones[v] = frecuentes[d.argmin()]

    return df.replace({col: correcciones})


    """Corrige dígitos duplicados en la edad"""
def arreglar_age(val):
    if pd.isna(val) or val == "" or str(val).strip().lower() in ["nan", "none"]:
        return np.nan

    s = str(val).strip()
    s_clean = s[:-2] if s.endswith(".0") else s

    if len(s_clean) == 3 and s_clean.isdigit():
        if s_clean[1] == s_clean[2]:
            return float(s_clean[:2])
        elif s_clean[0] == s_clean[1]:
            return float(s_clean[1:])
        else:
            return float(s_clean[:2])

    try:
        return float(s)
    except ValueError:
        return np.nan

    """Quita decimales innecesarios en edades enteras."""
def tipod_edad(val):
    if pd.isna(val) or str(val).strip().lower() in ["nan", "none", "", "null"]:
        return ""
    try:
        num = float(val)
        if num.is_integer():
            return int(num)
        return num
    except (ValueError, TypeError):
        return ""

    """Valida que la clase esté entre 1 y 3."""
def corregir_pclass(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    digitos = pd.Series(s).str.extract(r"(\d)")[0].values[0]
    if pd.notna(digitos):
        num = int(digitos)
        if 1 <= num <= 3:
            return num
    return np.nan

    """Convierte Survived a 0 o 1, corrigiendo O->0 y l->1."""
def limpiar_binario(val):
    if pd.isna(val):
        return np.nan

    s = (
        str(val).strip()
        .replace("I", "1").replace("l", "1")
        .replace("O", "0").replace("o", "0")
    )
    digitos = pd.Series(s).str.extract(r"(\d+)")[0].values[0]

    if pd.notna(digitos):
        num = int(digitos[0])
        if num in [0, 1]:
            return num
    return np.nan

    """Limpia SibSp y Parch, que son enteros pequeños."""
def limpiar_enteros_discretos(val):
    if pd.isna(val) or str(val).strip().lower() in ["nan", "none", "", "null"]:
        return np.nan

    s = (
        str(val).strip()
        .replace("I", "1").replace("l", "1")
        .replace("O", "0").replace("o", "0")
    )
    digitos = pd.Series(s).str.extract(r"(\d+)")[0].values[0]

    if pd.notna(digitos):
        # '33' con todos los dígitos iguales es un 3 duplicado
        if len(digitos) > 1 and len(set(digitos)) == 1:
            return int(digitos[0])
        return int(digitos)
    return np.nan

    """Quita símbolos de moneda y normaliza el separador decimal de la tarifa."""
def corregir_fare(val):

    if pd.isna(val) or str(val).strip().lower() in ["nan", "none", "", "null"]:
        return np.nan
    s = str(val).strip()

    # Deja solo dígitos, separadores y letras que imitan dígitos
    s = re.sub(r"[^\d.,OoSs]", "", s)
    s = s.replace("O", "0").replace("o", "0").replace("S", "5").replace("s", "5")

    if "," in s and "." in s:
        s = s.replace(",", "")      
    elif "," in s:
        s = s.replace(",", ".")     

    try:
        num = float(s)
    except ValueError:
        return np.nan

    while num > FARE_MAX:
        num /= 10.0

    return round(num, 4)

    """Restaura el punto decimal usando el rango típico de cada clase como referencia."""
def ajustar_por_clase(fila):

    fare = fila["Fare"]
    clase = fila["Pclass"]

    if pd.isna(fare) or pd.isna(clase):
        return fare
    tope = TOPE_POR_CLASE.get(int(clase), FARE_MAX)

    while fare > tope:
        fare /= 10.0
    return round(fare, 4)

    """Unifica la escritura del prefijo del ticket."""
def estandarizar_ticket(val):
    if pd.isna(val) or str(val).strip().lower() in ["nan", "none", "", "null"]:
        return np.nan
    s = re.sub(r"\s+", " ", str(val).strip().upper())

    # Separa prefijo y número.
    m = re.match(r"^(.*?)\s*(\d+)$", s)

    # Sin número al final (ej. 'LINE'): se devuelve tal cual
    if m is None:
        return s

    prefijo, numero = m.group(1), m.group(2)

    # Ticket puramente numérico
    if prefijo == "":
        return numero

    # Clave: solo letras y dígitos, sin puntos ni barras
    clave = re.sub(r"[^A-Z0-9]", "", prefijo)

    # si no está en el diccionario, deja el prefijo original
    estandar = PREFIJOS_ESTANDAR.get(clave, prefijo)

    return f"{estandar} {numero}"

    """Rellena vacíos eligiendo al azar entre los valoresexistentes, con probabilidad proporcional a su frecuencia."""
def rellenar_proporcional(df, columna):

    porcentajes = df[columna].value_counts(normalize=True)

    if DIAGNOSTICO:
        print(porcentajes)

    vacios = df[columna].isna()
    if vacios.sum() == 0:
        return df

    df.loc[vacios, columna] = np.random.choice(
        porcentajes.index,
        size=vacios.sum(),
        p=porcentajes.values,
    )
    return df

# 7. LIMPIEZA POR COLUMNAS
for col in ["Sex", "Embarked"]:
    df = limpiar(df, col)

df["Embarked"] = df["Embarked"].replace({"Ss": "S", "Ss.": "S"})

# 8. COLUMNAS NUMERICAS

df["PassengerId"] = df["PassengerId"].astype(str).str.extract(r"(\d+)")[0]

df["Pclass"] = df["Pclass"].apply(corregir_pclass)
df["Pclass"] = df["Pclass"].fillna(df["Pclass"].mode()[0]).astype(int)

# Normalización tipográfica
for col in ["Age", "Fare", "SibSp", "Parch", "Survived"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace("O", "0", case=False)
        .str.replace("S", "5", case=False)
        .str.replace(",", ".")
    )

df["Age"] = df["Age"].apply(arreglar_age).apply(tipod_edad)

df["Survived"] = df["Survived"].apply(limpiar_binario)
df["Survived"] = df["Survived"].fillna(df["Survived"].mode()[0]).astype(int)

df["SibSp"] = df["SibSp"].apply(limpiar_enteros_discretos).astype("Int64")
df["Parch"] = df["Parch"].apply(limpiar_enteros_discretos).astype("Int64")

# Descarta filas sin ID ni nombre
df = df[
    df["PassengerId"].notna()
    & df["Name"].notna()
    & (df["Name"].str.strip() != "")
].reset_index(drop=True)

# Normalizar Titulo

MAPA_TITULOS = {
    # variantes de Mr
    "MMr": "Mr", "Mrr": "Mr", "rM": "Mr", "Msr": "Mr",
    "r": "Mr", "M": "Mr",
    # variantes de Mrs
    "Mrrs": "Mrs", "rs": "Mrs",
    # variantes de Miss
    "Miiss": "Miss", "iMss": "Miss", "Mis": "Miss",
    "iss": "Miss", "s": "Miss",
    # variantes de Master
    "Mastir": "Master",
    # equivalencias de idioma del dataset original
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
}

df["Titulo"] = df["Titulo"].replace(MAPA_TITULOS)

print(df["Titulo"].value_counts(dropna=False).to_string())


# 9. Fare
#   corregir formato
#   ajustar el punto decimal por clase
#   rellenar con la mediana

df["Fare"] = df["Fare"].apply(corregir_fare)

# axis=1 aplica la función FILA por fila (necesita Fare y Pclass juntas)
df["Fare"] = df.apply(ajustar_por_clase, axis=1)
df["Fare"] = df.groupby("Pclass")["Fare"].transform(lambda x: x.fillna(x.median()))

# 9.2. RELLENAR Age
# Los vacíos se rellenan con la mediana de edad de su título
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Age_imputada"] = df["Age"].isna()

# groupby divide por Titulo
df["Age"] = df.groupby("Titulo")["Age"].transform(lambda x: x.fillna(x.median()))

df["Age"] = df["Age"].fillna(df["Age"].median())

# Verificar
print("nulos en Age:", df["Age"].isna().sum())
print("imputadas:", df["Age_imputada"].sum())
print("\nmediana por título:")
print(df.groupby("Titulo")["Age"].median().to_string())


# 10. Ticket

df["Ticket"] = df["Ticket"].apply(estandarizar_ticket)
df["Ticket"] = df["Ticket"].fillna("Desconocido")

if DIAGNOSTICO:
    # \D+ = uno o más caracteres que NO son dígitos
    prefijos = df["Ticket"].str.extract(r"^(\D+)")[0].str.strip()
    print(prefijos.value_counts(dropna=True).to_string())

# 11. Cabin y Deck

df["Cabin"] = df["Cabin"].astype(str).str.strip().str.upper()
df["Cabin"] = df["Cabin"].replace(
    {"NAN": np.nan, "NONE": np.nan, "": np.nan, "333": np.nan}
)
df.loc[df["Cabin"] == "CC 65", "Cabin"] = "C65"

df["Deck"] = df["Cabin"].str.extract(r"([A-T])")[0]

df["Cabin"] = df["Cabin"].fillna("Desconocido")
df["Deck"] = df["Deck"].fillna("Desconocido")

# 12. RELLENAR Parch, Embarked y SibSp DE FORMA PROPORCIONAL

for col in ["Parch", "Embarked", "SibSp"]:
    df = rellenar_proporcional(df, col)

# COLUMNAS DERIVADAS PARA EL ANÁLISIS
df["TamanoFamilia"] = df["SibSp"] + df["Parch"] + 1
df["ViajaSolo"] = (df["TamanoFamilia"] == 1)
df["GrupoEdad"] = pd.cut(
    df["Age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Niño", "Adolescente", "Adulto joven", "Adulto", "Mayor"],
    right=False,
)
personas_por_ticket = df.groupby("Ticket")["PassengerId"].transform("count")
df["TarifaPorPersona"] = (df["Fare"] / personas_por_ticket).round(4)

df["Puerto"] = df["Embarked"].map({
    "S": "Southampton", "C": "Cherburgo", "Q": "Queenstown"
})

df["Estado"] = df["Survived"].map({0: "Falleció", 1: "Sobrevivió"})

# REDONDEO 
#  Age 

df["Age"] = np.where(
    df["Age"] < 1,        
    1,                    
    df["Age"].round(0)    
)

df["Age"] = df["Age"].astype("Int64")


#  Fare y TarifaPorPersona 
df["Fare"] = df["Fare"].round(0).astype("Int64")
df["TarifaPorPersona"] = df["TarifaPorPersona"].round(0).astype("Int64")


#  Verificar 
print("Age  → min:", df["Age"].min(), "| max:", df["Age"].max())
print("Fare → min:", df["Fare"].min(), "| max:", df["Fare"].max())
print("\ntipos de las columnas numéricas:")
print(df[["Age", "Fare", "TarifaPorPersona", "SibSp",
          "Parch", "TamanoFamilia"]].dtypes)


COLUMNAS_AUXILIARES = ["Titulo", "Apellido", "Deck", "Age_imputada"]

df_salida = df.drop(columns=COLUMNAS_AUXILIARES, errors="ignore")

df_salida.to_csv("Titanic_limpio.csv", index=False, encoding="utf-8-sig")
df.to_csv("Titanic_powerbi.csv", index=False, sep=";", encoding="utf-8-sig")
print("columnas exportadas:", df_salida.columns.tolist())

# RESUMEN FINAL

print(f"filas iniciales : {filas_iniciales}")
print(f"filas finales   : {len(df)}")
print(f"IDs distintos   : {df['PassengerId'].nunique()}")
print(f"nulos totales   : {df.isna().sum().sum()}")
print(f"columnas        : {df.columns.tolist()}")
print()
print(df.groupby("Pclass")["Fare"].describe()[["min", "50%", "max"]])