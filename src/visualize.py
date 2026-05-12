

epochs = [1, 2, 3]
loss = [0.2058, 0.0449, 0.0211]
auc = [0.9964, 0.9978, 0.9977]

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotting Loss
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss (BCE)', color=color)
ax1.plot(epochs, loss, marker='o', color=color, label='Loss')
ax1.tick_params(axis='y', labelcolor=color)

# Plotting AUC on a second y-axis
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Validation ROC-AUC', color=color)
ax2.plot(epochs, auc, marker='s', color=color, label='ROC-AUC')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Training Loss vs. Validation ROC-AUC')
plt.grid(True, linestyle='--')
plt.savefig('training_metrics.png')
plt.show()