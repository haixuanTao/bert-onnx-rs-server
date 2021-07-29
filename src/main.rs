use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use onnxruntime::*;
use serde::Deserialize;
use std::sync::Mutex;
use tokenizers::models::wordpiece::WordPieceBuilder;
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::tokenizer::AddedToken;
use tokenizers::tokenizer::{EncodeInput, Tokenizer};
use tokenizers::utils::padding::{PaddingDirection::Right, PaddingParams, PaddingStrategy::Fixed};
#[derive(Deserialize)]
struct DataQuery {
    data: String,
}

async fn use_onnx(state: web::Data<AppState>, query: web::Query<DataQuery>) -> impl Responder {
    let input = state
        .tokenizer
        .encode(EncodeInput::Single(query.data.clone()), true)
        .unwrap();

    let _mask: Vec<i64> = input
        .get_attention_mask()
        .iter()
        .map(|x| *x as i64)
        .collect();
    let mask = ndarray::Array::from_vec(_mask).into_shape((1, 60)).unwrap();

    let _token: Vec<i64> = input.get_ids().iter().map(|x| *x as i64).collect();
    let token = ndarray::Array::from_vec(_token)
        .into_shape((1, 60))
        .unwrap();

    let mut session = state.session.lock().unwrap();
    let _outputs: Vec<tensor::OrtOwnedTensor<f32, _>> = session.run(vec![token, mask]).unwrap();
    HttpResponse::Ok().body(format!("{}: {}", query.data, _outputs[0].to_string(),))
}
// This struct represents state
struct AppState {
    tokenizer: Tokenizer,
    session: Mutex<onnxruntime::session::Session>,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(move || {
        let environment = environment::Environment::builder()
            .with_name("test")
            .build()
            .unwrap();
        let vocab_path = "./src/vocab.txt";
        let wp_builder = WordPieceBuilder::new()
            .files(vocab_path.into())
            .continuing_subword_prefix("##".into())
            .max_input_chars_per_word(100)
            .unk_token("[UNK]".into())
            .build()
            .unwrap();

        let mut tokenizer = Tokenizer::new(Box::new(wp_builder));
        tokenizer.with_padding(Some(PaddingParams {
            strategy: Fixed(60),
            direction: Right,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".into(),
        }));
        tokenizer.with_pre_tokenizer(Box::new(BertPreTokenizer));
        tokenizer.with_post_processor(Box::new(BertProcessing::new(
            ("[SEP]".into(), 102),
            ("[CLS]".into(), 101),
        )));
        tokenizer.with_normalizer(Box::new(BertNormalizer::new(true, true, false, false)));
        tokenizer.add_special_tokens(&[
            AddedToken {
                content: "[PAD]".into(),
                single_word: false,
                lstrip: false,
                rstrip: false,
            },
            AddedToken {
                content: "[CLS]".into(),
                single_word: false,
                lstrip: false,
                rstrip: false,
            },
            AddedToken {
                content: "[SEP]".into(),
                single_word: false,
                lstrip: false,
                rstrip: false,
            },
            AddedToken {
                content: "[MASK]".into(),
                single_word: false,
                lstrip: false,
                rstrip: false,
            },
        ]);
        let state = AppState {
            tokenizer,
            session: Mutex::new(
                environment
                    .new_session_builder()
                    .unwrap()
                    .use_cuda(0)
                    .unwrap()
                    .with_model_from_file("./src/onnx_model.onnx")
                    .unwrap(),
            ),
        };

        App::new().data(state).route("/", web::get().to(use_onnx))
    })
    .bind(("127.0.0.1", 8080))?
    .workers(1)
    .run()
    .await
}
