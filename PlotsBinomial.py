from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(1, 1)

n, p = 100, 0.64
mean, var, skew, kurt = binom.stats(n, p, moments='mvsk')
x = np.arange(binom.ppf(0.0001, n, p),
              binom.ppf(0.9999, n, p))
y=binom.pmf(x, n, p)
lc = binom.ppf(0.05, n, p)/n
uc = binom.ppf(0.975, n, p)/n
print(f"{lc}-{uc}")
lcbx = np.arange(binom.ppf(0.001, n, p),binom.ppf(0.027, n, p))
lcby=binom.pmf(lcbx, n, p)
ucbx = np.arange(binom.ppf(0.974, n, p),binom.ppf(0.999, n, p))
ucby=binom.pmf(ucbx, n, p)
print()
lw=3
ax.plot(x/n, y, 'b.', ms=8)
ax.vlines(x/n, 0, y, colors='b', lw=lw, alpha=0.1)
ax.vlines(lcbx/n, 0, lcby, colors='r', lw=lw, alpha=0.5)
#ax.vlines(ucbx/n, 0, ucby, colors='g', lw=lw, alpha=0.5)
plt.xlabel("Probability of Success")
plt.ylabel("Probability Mass")
plt.title('Probability Mass Function')

rv = binom(n, p)
#ax.vlines(x/n, 0, rv.pmf(x), colors='k', linestyles='-', lw=1,
#        label='frozen pmf')
ax.legend(loc='best', frameon=False)
plt.grid(True)
#plt.ylim([0,0.042])
#plt.xlim([0.4,.6])

fig.savefig(f"PMF_deltaLessThanPoint1.pdf", format='pdf', dpi=5000)
plt.show()
