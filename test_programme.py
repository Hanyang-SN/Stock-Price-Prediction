import pandas as pd
import matplotlib.pyplot as plt

# 데이터프레임 생성
df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 5, 8, 3, 6]})

# 선 그래프 데이터프레임 df
df = df["x"].diff(1)

plt.plot(df)
plt.show()

data = {"name": ["John", "Mike", "Sarah", "Jessica"], "score": [90, 80, 70, 85]}

data = data["name"]
print(data)
