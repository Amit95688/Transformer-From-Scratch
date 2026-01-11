import torch
from flask import Flask, render_template, request
from tokenizers import Tokenizer

from config import get_config
from model import build_transformer
from dataset import casual_mask

app = Flask(__name__)

config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizers
tokenizer_src = Tokenizer.from_file(
    config["tokenizer_file"].format(lang=config["source_lang"])
)
tokenizer_tgt = Tokenizer.from_file(
    config["tokenizer_file"].format(lang=config["target_lang"])
)

# Build model
model = build_transformer(
    src_vocab_size=len(tokenizer_src.get_vocab()),
    tgt_vocab_size=len(tokenizer_tgt.get_vocab()),
    src_seq_len=config["seq_length"],
    tgt_seq_len=config["seq_length"],
    d_model=config["d_model"],
    d_ff=config["dim_feedforward"],
    num_heads=config["nhead"],
    num_encoder_layers=config["num_encoder_layers"],
    num_decoder_layers=config["num_decoder_layers"],
    dropout=config["dropout"],
).to(device)

checkpoint = torch.load("model.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def translate(text: str) -> str:
    sos = tokenizer_src.token_to_id("[SOS]")
    eos = tokenizer_src.token_to_id("[EOS]")
    pad = tokenizer_src.token_to_id("[PAD]")

    tokens = tokenizer_src.encode(text).ids[: config["seq_length"] - 2]
    encoder_input = torch.tensor([[sos] + tokens + [eos]], device=device)
    encoder_mask = (encoder_input != pad).unsqueeze(1).unsqueeze(1)

    with torch.no_grad():
        encoder_output = model.encode(encoder_input, encoder_mask)
        decoder_input = torch.tensor([[tokenizer_tgt.token_to_id("[SOS]")]], device=device)

        for _ in range(config["seq_length"]):
            decoder_mask = casual_mask(decoder_input.size(1)).to(device)
            out = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
            logits = model.project(out)
            next_token = logits[:, -1].argmax(dim=-1).item()
            decoder_input = torch.cat(
                [decoder_input, torch.tensor([[next_token]], device=device)], dim=1
            )
            if next_token == tokenizer_tgt.token_to_id("[EOS]"):
                break

    return tokenizer_tgt.decode(decoder_input.squeeze(0).tolist())


@app.route("/", methods=["GET", "POST"])
def index():
    translated_text = None
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("text", "")
        if input_text:
            translated_text = translate(input_text)

    return render_template(
        "index.html",
        input_text=input_text,
        translated_text=translated_text,
        device=device.type,
    )


if __name__ == "__main__":
    app.run(debug=True)