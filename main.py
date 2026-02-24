# ... all imports from previous main.py ...
from save_and_push import save_model_locally, push_model_to_hub

def main():
    # 1️⃣ Download dataset
    download_and_extract_dataset()

    # 2️⃣ Set seeds & check GPU
    set_seed(42)
    device = check_gpu()

    # 3️⃣ Load training data
    sentences, labels = load_dataset()
    input_ids, attention_masks, tokenizer = tokenize_and_pad(sentences)
    train_inputs, val_inputs, train_labels, val_labels, train_masks, val_masks = split_data(input_ids, attention_masks, labels)

    # 4️⃣ Create dataloaders
    batch_size = 32
    train_dataloader = create_dataloader(train_inputs, train_masks, train_labels, batch_size=batch_size, mode='train')
    validation_dataloader = create_dataloader(val_inputs, val_masks, val_labels, batch_size=batch_size, mode='val')

    # 5️⃣ Build model, optimizer, scheduler
    model = build_model(num_labels=2)
    model.to(device)
    epochs = 4
    optimizer, scheduler = prepare_optimizer_scheduler(model, train_dataloader, epochs=epochs)

    # 6️⃣ Training loop
    loss_values = []
    for epoch in range(epochs):
        print(f"\n======== Epoch {epoch+1} / {epochs} ========")
        t0 = time.time()
        avg_train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        loss_values.append(avg_train_loss)
        print(f"  Average training loss: {avg_train_loss:.2f}")
        print(f"  Training epoch took: {format_time(time.time() - t0)}")

    # 7️⃣ Plot training loss
    plot_training_loss(loss_values)

    # 8️⃣ Load test set & evaluate
    from pandas import read_csv
    df_test = read_csv("./cola_public/raw/out_of_domain_dev.tsv", delimiter='\t', header=None,
                       names=['sentence_source','label','label_notes','sentence'])
    test_sentences = df_test.sentence.values
    test_labels = df_test.label.values

    test_dataloader = prepare_test_dataloader(test_sentences, test_labels, tokenizer, max_len=64, batch_size=batch_size)
    mcc = evaluate_model(model, test_dataloader, device)
    print(f"\nTest set Matthews Correlation Coefficient (MCC): {mcc:.3f}")

    # 9️⃣ Save locally and optionally push to Hugging Face Hub
    local_dir = save_model_locally(model, tokenizer, output_dir='./model_save/')
    # push_model_to_hub(model, tokenizer, repo_name="ml-dl-ops-model")  # uncomment to push

if __name__ == "__main__":
    main()
