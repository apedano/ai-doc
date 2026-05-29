Example:

| Type                        | LATEX markup                                                | Renders as                                                  |
|-----------------------------|-------------------------------------------------------------|-------------------------------------------------------------|
| Plain                       | `\begin{matrix}   1 & 2 & 3\\   a & b & c   \end{matrix}`   | $\begin{matrix}   1 & 2 & 3\\   a & b & c   \end{matrix}$   |
| Parentheses; round brackets | `\begin{pmatrix}   1 & 2 & 3\\   a & b & c   \end{pmatrix}` | $\begin{pmatrix}   1 & 2 & 3\\   a & b & c   \end{pmatrix}$ |
| Brackets; square brackets   | `\begin{bmatrix}   1 & 2 & 3\\   a & b & c   \end{bmatrix}` | $\begin{bmatrix}   1 & 2 & 3\\   a & b & c   \end{bmatrix}$ |
| Braces; curly brackets      | `\begin{Bmatrix}   1 & 2 & 3\\   a & b & c   \end{Bmatrix}` | $\begin{Bmatrix}   1 & 2 & 3\\   a & b & c   \end{Bmatrix}$ |
| Pipes                       | `\begin{vmatrix}   1 & 2 & 3\\   a & b & c   \end{vmatrix}` | $\begin{vmatrix}   1 & 2 & 3\\   a & b & c   \end{vmatrix}$ |
| Double pipes                | `\begin{Vmatrix}   1 & 2 & 3\\   a & b & c   \end{Vmatrix}` | $\begin{Vmatrix}   1 & 2 & 3\\   a & b & c   \end{Vmatrix}$ |

With dots

```
\begin{bmatrix}
1 & 2 & \cdots & n \\
2 & 4 & \cdots & 2n \\
\vdots & \vdots & \ddots & \vdots \\
n & 2n & \cdots & n^2 \\
\end{bmatrix}
```

$$
\begin{bmatrix}
1 & 2 & \cdots & n \\
2 & 4 & \cdots & 2n \\
\vdots & \vdots & \ddots & \vdots \\
n & 2n & \cdots & n^2 \\
\end{bmatrix}
$$