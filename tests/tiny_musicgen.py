"""Build a tiny, randomly initialised MusicGen checkpoint on disk.

Used to exercise the real transformers MusicGen code path (processor,
generate, audio prompts, decoding, saving) without downloading weights. The
audio it produces is noise; tests only check shapes, rates and plumbing.
"""

from __future__ import annotations

from pathlib import Path


def build_tiny_musicgen(dest: Path, *, melody: bool = False) -> Path:
    import transformers as tf
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = ["<pad>", "</s>", "<unk>", "ambient", "piano", "music", "drums", "dark", "synth"]
    tok = Tokenizer(models.Unigram([(w, -1.0) for w in words], unk_id=2))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = tf.T5TokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="</s>",
                                   unk_token="<unk>", extra_ids=0)
    feature_extractor = tf.EncodecFeatureExtractor(feature_size=1, sampling_rate=32000)
    text = tf.T5Config(vocab_size=len(words), d_model=16, d_kv=4, d_ff=32, num_layers=1,
                       num_heads=2, decoder_start_token_id=0)
    audio = tf.EncodecConfig(sampling_rate=32000, audio_channels=1, num_filters=4,
                             hidden_size=8, codebook_size=32, codebook_dim=8,
                             upsampling_ratios=[8, 5, 4, 4], num_lstm_layers=1,
                             target_bandwidths=[1.0], norm_type="weight_norm",
                             use_causal_conv=False)
    decoder_kwargs = dict(vocab_size=32, hidden_size=16, num_hidden_layers=1,
                          num_attention_heads=2, ffn_dim=32, num_codebooks=4,
                          max_position_embeddings=2048, audio_channels=1,
                          pad_token_id=32, bos_token_id=32)
    if melody:
        decoder = tf.MusicgenMelodyDecoderConfig(**decoder_kwargs)
        config = tf.MusicgenMelodyConfig(text_encoder=text.to_dict(),
                                         audio_encoder=audio.to_dict(),
                                         decoder=decoder.to_dict(), num_chroma=12)
        model = tf.MusicgenMelodyForConditionalGeneration(config)
        fe = tf.MusicgenMelodyFeatureExtractor(sampling_rate=32000)
        processor = tf.MusicgenMelodyProcessor(feature_extractor=fe, tokenizer=tokenizer)
    else:
        decoder = tf.MusicgenDecoderConfig(**decoder_kwargs)
        config = tf.MusicgenConfig(text_encoder=text.to_dict(), audio_encoder=audio.to_dict(),
                                   decoder=decoder.to_dict())
        model = tf.MusicgenForConditionalGeneration(config)
        processor = tf.MusicgenProcessor(feature_extractor=feature_extractor,
                                         tokenizer=tokenizer)
    model.generation_config.decoder_start_token_id = 32
    model.generation_config.pad_token_id = 32
    model.generation_config.max_length = 1500
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest)
    processor.save_pretrained(dest)
    return dest
