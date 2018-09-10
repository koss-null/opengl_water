# opengl_water
learning to opengl (math modelling)

This project is written to get a simple guesses about how opengs works.
As the result there is a water surface model with some features:
- you can see sky and bed reflections
- you can change position of camera view
- you can place the camera under the water (use wsad)
- bed could be not flat (breakdowns are well even when the surface angle changing is tremendous)
- it's provided a couple of strategies to immulate waves (use space to sturt waving)
- the deeper it is, the less red light reflects
- there are some good refractions on the moving water 

To run the program it's possible to use just
  cd src/my_water
  python3 main.py
