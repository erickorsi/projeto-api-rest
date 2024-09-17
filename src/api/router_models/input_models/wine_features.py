from pydantic import BaseModel

class Features(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyans: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float
