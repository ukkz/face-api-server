//import '@tensorflow/tfjs-node';
const fs = require('fs');
const path = require('path');
const canvas = require('canvas');
const faceapi = require('face-api.js');
const express = require('express');
const multer = require('multer');

// Express App
const app = express();
// ポート
const port = process.env.PORT || 8000;

// 顔認識モデル
const faceDetectionNet = faceapi.nets.ssdMobilenetv1;
const minConfidence = 0.5;
const faceDetectionOptions = new faceapi.SsdMobilenetv1Options({ minConfidence });

// HTML Canvasの代わり（画像処理する部分）
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

// 動的に生成される画像（アップロードと認識後）が保存されるディレクトリ
const save_img_dir = path.resolve(__dirname, './img');
// 存在確認をしてなければ作成（Herokuデプロイ等のときしておかなければENOENTでエラー）
if (!fs.existsSync(save_img_dir)) {
    fs.mkdirSync(save_img_dir);
}
// 指定ディレクトリ以下の画像ファイルに直接アクセスできるようにする
app.use(express.static('img'));

// 解析後の画像保存パス（ファイル名固定）
// ここにアクセスすると解析後画像を取得できる
const out_img_file = path.resolve(save_img_dir, 'result.jpg');

// 解析前の画像（アップロードされたやつ）が保存されるパス（ファイル名固定）
const storage = multer.diskStorage({
    destination: function (req, file, callback) {
        // 保存ディレクトリ
        callback(null, save_img_dir);
    },
    filename: function (req, file, callback) {
        // ファイル名
        callback(null, 'original.jpg');
    }
});
const upload = multer({ storage: storage });

// あらかじめweightsを読み込んでおく
(async function () {
    await faceDetectionNet.loadFromDisk('./weights');
    await faceapi.nets.faceExpressionNet.loadFromDisk('./weights');
    console.log('Ready on port', port);
}());


// 顔認識実行
async function run(target_filename) {
    // 画像を読み込む
    const img = await canvas.loadImage(target_filename);
    // 顔認識を開始（位置検出と表情認識）
    // 赤ちゃん1人ならdetectSingleFaceのほうが軽いかもしれません
    console.log('Face recognition running...');
    const results = await faceapi.detectAllFaces(img, faceDetectionOptions).withFaceExpressions();
    // 画像からCanvasを作成
    const out = faceapi.createCanvasFromMedia(img);
    // 検出した顔に枠を書き込む
    faceapi.draw.drawDetections(out, results);
    // 表情のうち最も係数が大きいものを書き込む
    faceapi.draw.drawFaceExpressions(out, results);
    // 保存
    fs.writeFileSync(out_img_file, out.toBuffer('image/jpeg'));
    console.log('Done!');
    // 結果のオブジェクトを返す
    return results;
}


// APIサーバーを起動
// http://host:8000/ に、form-dataで "file" をキーとして画像をPOSTすればOK
const server = app.listen(port);
app.post('/', upload.single('file'), async function(req, res, next) {
    // POSTされた画像の情報をJSONで取得
    const file_info = JSON.stringify(req.file);
    // アップロードされたファイル名（オリジナル名）を確認
    if (req.file.originalname) {
        console.log('Uploaded', req.file.originalname);
    } else {
        console.log('Uploaded file (originalname unknnown)');
    }
    

    // サーバーに保存後の画像ファイル名（固定だけど）を指定して顔認識を行う
    // awaitのためここでブロッキングに注意
    const recognition = await run(path.resolve(save_img_dir, req.file.filename));

    // 結果を返す
    res.json(recognition);
});
