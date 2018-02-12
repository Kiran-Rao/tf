# Some playing around with Keras

https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html


Notes for on optimization performance:

All training done with 10000 mini batches of 50

2x Dense Networks (128): 0.95
Increase layer density (256): no diff
Add dropout (0.5): 0.97
Add additional layer: no diff
Change training rate to 0.1: Slightly slower, but more stable error: ~0.975
Use Momentum Optimizer (0.1, 0.5): Much faster training, 0.98

Use Momentum(0.1, 0.5) for 5000, Momentum(0.05, 0.5): 0.97~

Use Momentum(0.1, 0.5) for 5000, GradientDescent(0.05) for 5000
- Much more stable, but still only getting 0.98

Moved to 256, got 0.983
3 layers of 128 got 0.982
