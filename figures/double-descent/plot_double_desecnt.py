import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("ib-dd-fashionmnist-fcn-test.csv")
df = df.sort_values(by=["total_params"])
total_params = df["total_params"]
train_error = 1 - df["avg_train_acc"]
test_error = 1 - df["avg_test_acc"]
train_loss = df["avg_train_loss"]
test_loss = df["avg_test_loss"]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(total_params, train_error, label="train")
ax[0].plot(total_params, test_error, label="test")
ax[0].axvline(60000, color="red", linestyle="--")
ax[0].set_xlabel("total_params")
ax[0].set_ylabel("error")

ax[1].plot(total_params, train_loss, label="train")
ax[1].plot(total_params, test_loss, label="test")
ax[1].axvline(60000, color="red", linestyle="--")
ax[1].set_xlabel("total_params")
ax[1].set_ylabel("loss")

ax[0].legend()
ax[1].legend()
plt.savefig("double-descent-fashionmnist-fcn.pdf")
