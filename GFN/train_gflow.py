from gflownet import function_to_tensor, face_parents, sorted_keys
from reward import function_reward
import tqdm
import torch
from torch.distributions.categorical import Categorical



def train_gflow(F_sa, train_loader, val_loader, opt,  num_episodes=50000, update_freq=4):
        """Train the GFlow model."""
        losses = []
        sampled_faces = []
        minibatch_loss = 0
        
        for episode in tqdm.tqdm(range(num_episodes), ncols=40):
            state = []
            edge_flow_prediction = F_sa(function_to_tensor(state))

            for t in range(3):
                policy = edge_flow_prediction / edge_flow_prediction.sum()
                action = Categorical(probs=policy).sample()  # Sample the action
                
                new_state = state + [sorted_keys[action]]
                parent_states, parent_actions = face_parents(new_state)
                px = torch.stack([function_to_tensor(p) for p in parent_states])
                pa = torch.tensor(parent_actions).long()

                aux = F_sa(px)
                parent_edge_flow_preds = aux[torch.arange(len(parent_states)), pa]

                if t == 2:
                    reward = function_reward(new_state, train_loader, val_loader)[1]
                    edge_flow_prediction = torch.zeros(6)
                else:
                    reward = 0
                    edge_flow_prediction = F_sa(function_to_tensor(new_state))

                flow_mismatch = (parent_edge_flow_preds.sum() - edge_flow_prediction.sum() - reward).pow(2)
                minibatch_loss += flow_mismatch  # Accumulate
                state = new_state

            sampled_faces.append(state)
            if episode % update_freq == 0:
                losses.append(minibatch_loss.item())
                minibatch_loss.backward()
                opt.step()
                opt.zero_grad()
                minibatch_loss = 0

        return losses, sampled_faces


def sample_gflow(F_sa, num_samples=1000):
    states_list = []

    for episode in tqdm.tqdm(range(1000), ncols=40):
        # Each episode starts with an "empty state"
        state = []
        # Predict F(s, a)
        edge_flow_prediction = F_sa(function_to_tensor(state))
        #print(edge_flow_prediction)

        for t in range(3):
            # The policy is just normalizing, and gives us the probability of each action
            policy = edge_flow_prediction / edge_flow_prediction.sum()
            #print(" ")
            #print(f"policy: {policy}")
            # Sample the action
            action = Categorical(probs=policy).sample()
            #print(f"action: {action}")
            # "Go" to the next state
            new_state = state + [sorted_keys[action]]
            #print(f"new_state: {new_state}")


            # Now we want to compute the loss, we'll first enumerate the parents
            parent_states, parent_actions = face_parents(new_state)
            #print(f"parent_states: {parent_states}")
            #print(f"parent_actions: {parent_actions}")
            # And compute the edge flows F(s, a) of each parent
            px = torch.stack([function_to_tensor(p) for p in parent_states])
            #print(f"px: {px}")
            pa = torch.tensor(parent_actions).long()
            #print(f"pa: {pa}")

            aux = F_sa(px)
            #print(f"aux: {aux}")
            parent_edge_flow_preds = aux[torch.arange(len(parent_states)), pa]
            #print(f"parent_edge_flow_preds: {parent_edge_flow_preds}")

            # Now we need to compute the reward and F(s, a) of the current state,
            # which is currently `new_state`
            if t == 2:
                # If we've built a complete face, we're done, so the reward is > 0
                # (unless the face is invalid)
                #print(new_state)
                reward = function_reward(new_state)[1]
                # and since there are no children to this state F(s,a) = 0 \forall a
                edge_flow_prediction = torch.zeros(6)
            else:
                # Otherwise we keep going, and compute F(s, a)
                reward = 0
                edge_flow_prediction = F_sa(function_to_tensor(new_state))

            # The loss as per the equation above
            flow_mismatch = (parent_edge_flow_preds.sum() - edge_flow_prediction.sum() - reward).pow(2)
            minibatch_loss += flow_mismatch  # Accumulate
            # Continue iterating
            state = new_state
        states_list.append(state)

    return states_list