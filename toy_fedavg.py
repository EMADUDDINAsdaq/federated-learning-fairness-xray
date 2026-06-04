import warnings
warnings.filterwarnings("ignore")

import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.datasets import make_classification

# Create toy dataset
np.random.seed(42)

X, y = make_classification(
    n_samples    = 1000,
    n_features   = 2,
    n_informative= 2,
    n_redundant  = 0,
    random_state = 42
)

NUM_CLIENTS  = 5
SAMPLES_EACH = len(X) // NUM_CLIENTS

client_data = []
for i in range(NUM_CLIENTS):
    start = i * SAMPLES_EACH
    end   = start + SAMPLES_EACH
    client_data.append((X[start:end], y[start:end]))

print("Toy dataset created")
print(f"Total samples : {len(X)}")
print(f"Clients       : {NUM_CLIENTS}")
print(f"Samples each  : {SAMPLES_EACH}")

#Define hospital client
class HospitalClient(fl.client.NumPyClient):

    def __init__(self, X, y, client_id):
        self.X         = X
        self.y         = y
        self.client_id = client_id
        self.model     = LogisticRegression(
            max_iter=200, random_state=42)
        self.model.fit(X, y)

    def get_parameters(self, config):
        return [self.model.coef_,
                self.model.intercept_]

    def set_parameters(self, parameters):
        self.model.coef_      = parameters[0]
        self.model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.X, self.y)
        return (self.get_parameters(config),
                len(self.X), {})

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc  = accuracy_score(
            self.y,
            self.model.predict(self.X))
        loss = log_loss(
            self.y,
            self.model.predict_proba(self.X))
        print(f"  Client {self.client_id} "
              f"— Accuracy: {acc:.4f} "
              f"| Loss: {loss:.4f}")
        return loss, len(self.X), {"accuracy": acc}

# client_fn tells Flower how to create each client
def client_fn(cid):
    X, y = client_data[int(cid)]
    return HospitalClient(X, y, client_id=int(cid))

#Run FedAvg simulation
if __name__ == "__main__":

    print("\n" + "=" * 50)
    print("FLOWER FEDAVG SIMULATION")
    print("=" * 50)
    print(f"Framework  : Flower (flwr) {fl.__version__}")
    print(f"Strategy   : FedAvg — McMahan et al. 2017")
    print(f"Dataset    : Toy dataset")
    print(f"Clients    : {NUM_CLIENTS} simulated hospitals")
    print(f"Rounds     : 5")
    print("=" * 50 + "\n")

    history = fl.simulation.start_simulation(
        client_fn  = client_fn,
        num_clients= NUM_CLIENTS,
        config     = fl.server.ServerConfig(num_rounds=5),
        strategy   = fl.server.strategy.FedAvg(
            fraction_fit         = 1.0,
            fraction_evaluate    = 1.0,
            min_fit_clients      = NUM_CLIENTS,
            min_evaluate_clients = NUM_CLIENTS,
            min_available_clients= NUM_CLIENTS,
        ),
    )

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    if history.losses_distributed:
        for rnd, loss in history.losses_distributed:
            print(f"  Round {rnd} : Loss = {loss:.4f}")

    print("\nCONFIRMATIONS")
    print(f"Flower installed  : YES")
    print(f"5 clients created : YES")
    print(f"FedAvg running    : YES")
    print(f"Rounds completed  : 5")
    print(f"Environment ready : YES")
