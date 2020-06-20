import torch
import torch.utils.data as data
from GaitSequenceDataset import GaitSequenceDataset
from preprocess import prepare_dataset,get_data_dimensions
from autoencoder import AE
from torch.nn import CrossEntropyLoss, MSELoss
from statistics import mean

def train_model(model, dataset, lr, epochs, logging):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters())
    # criterion = CrossEntropyLoss()
    criterion = MSELoss(size_average=False)

    for epoch in range(1, epochs + 1):
        model.train()

        if not epoch % 50:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr * (0.993 ** epoch)

        losses, embeddings = [], []
        for seq_true in dataset:
            optimizer.zero_grad()

            # Forward pass
            
            seq_true.to(device)
            seq_true=seq_true.float()
            seq_pred = model(seq_true)
            
            loss = criterion(seq_pred, seq_true)

            # Backward pass
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            embeddings.append(seq_pred)

        if logging:
            print("Epoch: {}, Loss: {}".format(str(epoch), str(mean(losses))))
        if epoch%10==0:
          torch.save([model.encoder, model.decoder],'autoencoder.pkl')
    return embeddings, mean(losses)

def encoding(train_dataset,encoding_dim,lr,epoch,logging=False):

  train_set, seq_len, num_features = get_data_dimensions(train_dataset)
  print(seq_len, num_features)
  model = AE(seq_len, num_features, encoding_dim)
  embeddings, f_loss = train_model(model, train_set, lr, epoch, logging  )

  return model.encoder, model.decoder, embeddings, f_loss

def main():
    lr=1e-3
    epochs=50

    dataset = GaitSequenceDataset(root_dir ='/home/shalini/Downloads/KL_Study_HDF5_for_learning/data/',
                                longest_sequence = 85,
                                shortest_sequence = 55)

    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True)
    train_dataset,test_dataset=prepare_dataset(dataloader)
    encoder, decoder, embeddings, f_loss = encoding(train_dataset,85,lr=lr,epoch=epochs,logging=True)

    torch.save([encoder,decoder],'autoencoder_final.pkl')
    print(f_loss)

    #test_set, seq_len, num_features = get_data_dimensions(test_dataset[0:1])
    test_encoding = encoder(test_dataset[0:1].float())
    test_decoding = decoder(test_encoding)

    print()
    print(test_encoding)
    print(test_decoding)

    

if __name__ == "__main__":
    main()