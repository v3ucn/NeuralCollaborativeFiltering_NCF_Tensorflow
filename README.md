# NeuralCollaborativeFiltering_NCF_Tensorflow

## How to use

Win/Linux
```
pip3 install tensorflow
```

Mac

```
pip install tensorflow-macos
```

```
python3 ncf_tensorflow.py
```

## output

```
学习前：
    User  Video 1  Video 2  Video 3  Video 4  Video 5  Video 6
0  User1     10.0      3.0      NaN      NaN      NaN      NaN
1  User2      NaN     10.0      NaN     10.0      5.0      1.0
2  User3      NaN      NaN      9.0      NaN      NaN      NaN
3  User4      6.0      1.0      NaN      8.0      NaN      9.0
4  User5      1.0      NaN      1.0      NaN     10.0      4.0
5  User6      1.0      4.0      1.0      NaN     10.0      1.0
6  User7      NaN      2.0      1.0      2.0      NaN      8.0
7  User8      NaN      NaN      NaN      1.0      NaN      NaN
8  User9      1.0      NaN     10.0      NaN      3.0      1.0


学习后：
        Video 1   Video 2   Video 3   Video 4   Video 5   Video 6
User1  9.508878  3.113009  7.157410  9.335443  9.629840  9.580980
User2  2.302773  9.624501  9.730996  9.773037  5.119677  0.930121
User3  3.733079  9.065267  9.348892  9.705484  8.523159  3.409261
User4  6.053867  1.170659  3.957999  7.933151  9.437999  8.902870
User5  1.106443  3.168726  0.732408  2.506290  9.728704  4.037748
User6  0.501248  4.096941  0.947260  2.387310  9.579254  1.118673
User7  4.283075  1.775985  0.756208  2.097234  9.486653  8.016829
User8  0.472433  1.816927  0.716490  1.029657  8.845912  1.527248
User9  1.068604  8.614190  9.599453  9.406795  3.018846  0.811254
```