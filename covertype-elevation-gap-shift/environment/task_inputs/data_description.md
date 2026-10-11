# Data description

Source: Covertype, UCI Machine Learning Repository (Blackard & Dean), CC BY 4.0, DOI 10.24432/C50K5N.
Each row is a 30 m x 30 m cell in Roosevelt National Forest, Colorado.

Files
- train.csv: `Id`, 54 feature columns, `Cover_Type` (1-7).
- test_features.csv: `Id` and the same 54 feature columns, no label.

Feature columns: Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology,
Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am,
Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points,
Wilderness_Area1..Wilderness_Area4 (binary), Soil_Type1..Soil_Type40 (binary).

Cover_Type: 1 Spruce/Fir, 2 Lodgepole Pine, 3 Ponderosa Pine, 4 Cottonwood/Willow,
5 Aspen, 6 Douglas-fir, 7 Krummholz.
