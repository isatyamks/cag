import matplotlib.pyplot as plt

class LivePlotter:
    def __init__(self, cag_setup_cost: int):
        self.cag_setup_cost = cag_setup_cost
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.fig.canvas.manager.set_window_title("CAG vs RAG Token Usage")

        self.queries = [0]
        self.cag_tokens = [self.cag_setup_cost]
        self.rag_tokens = [0]

        self.line_cag, = self.ax.plot(self.queries, self.cag_tokens, 'o-', color='#00d2ff', label='CAG Cumulative', linewidth=2)
        self.line_rag, = self.ax.plot(self.queries, self.rag_tokens, 'o-', color='#ffbb00', label='RAG Cumulative', linewidth=2)

        self.ax.set_xlabel("Query #", fontweight='bold')
        self.ax.set_ylabel("Total GPU Tokens Encoded", fontweight='bold')
        self.ax.set_title("Forward-Pass Workload Over Time", fontweight='bold')
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.ax.legend(loc='upper left')

        self.ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.tight_layout()
        plt.pause(0.1)

    def update(self, query_num: int, total_cag_new: int, total_rag: int):
        self.queries.append(query_num)
        self.cag_tokens.append(self.cag_setup_cost + total_cag_new)
        self.rag_tokens.append(total_rag)

        self.line_cag.set_data(self.queries, self.cag_tokens)
        self.line_rag.set_data(self.queries, self.rag_tokens)

        self.ax.relim()
        self.ax.autoscale_view()

        self.ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.draw()
        plt.pause(0.05)
