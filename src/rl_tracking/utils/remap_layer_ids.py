import pandas as pd

def make_remapping_csv(hits_df):
    #volume_order = [8, 7, 9, 13, 12, 14, 16, 17, 18]
    volume_order = [8, 13, 17, 7, 9, 12, 14, 16, 18]
    reverse_volumes = {7, 12, 16}
    unique_layers = hits_df[['volume_id', 'layer_id']].drop_duplicates()
    remapping = []
    layer_counter = 1
    for vol in volume_order:
        layers = sorted(unique_layers[unique_layers['volume_id'] == vol]['layer_id'].unique())
        if vol in reverse_volumes:
            layers = layers[::-1]
        for layer in layers:
            remapping.append([vol, layer, layer_counter])
            layer_counter += 1
    remapping = pd.DataFrame(remapping, columns = ['volume_id', 'layer_id', 'unique_layer_id'])
    remapping.to_csv('tml_layer_remap.csv')
    return

if __name__ == "__main__":
    hits_df = pd.read_csv('/Users/liv/trackML/train_1/event000002059-hits.csv')
    make_remapping_csv(hits_df)