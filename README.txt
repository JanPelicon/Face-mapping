Navodila za zagon:
Projekt vsebuje dve datoteki: 

project_two_camera.py (Preko argumentov moramo podati indeksa na katerih OpenCV najde kamere npr.: python project_two_camera.py 0 700)
project_one_camera.py (Preko argumentov moramo podati indeksa na katerih OpenCV najde kamero npr.: python project_two_camera.py 0)

Indekse kamer pridobimo, če poženemo skripto get_camera_index.py

V kolikor nimate cython-a ne na vašemu sistemu:
pip install cython

Zgraditi moramo Python knjižnico za napisane C funkcionalnosti.
python setup.py build_ext --inplace

####################
# CELOTEN POSTOPEK #
####################

get_camera_index.py (pridobimo X1 in X2)
python setup.py build_ext --inplace

project_one_camera.py X1
OR
project_two_camera.py X1 X2




