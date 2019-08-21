/** Thanks to Va Barbosa (https://github.com/vabarbosa) */

const modelUrl = '../model_web/model.json'
const labelsUrl = '../model_web/labels.json'

const imageSize = 400

let targetSize = { w: imageSize, h: imageSize }
let model
let imageElement
let videoElement
let webcamStream
let videoCanvas
let context
let inputTensor
let labels

/**
 * load the TensorFlow.js model
 */
const loadModel = async function () {
  message('loading model...')

  let start = (new Date()).getTime()

  // https://js.tensorflow.org/api/1.1.2/#loadGraphModel
  model = await tf.loadGraphModel(modelUrl)

  let end = (new Date()).getTime()

  message(model.modelUrl)
  message(`model loaded in ${(end - start) / 1000} secs`, true)
}

/**
 * handle image upload
 *
 * @param {HTMLInputElement} input - the image file upload element
 */
const loadImage = function (input) {
  if (input.files && input.files[0]) {
    message('resizing image...')

    let reader = new FileReader()

    reader.onload = function (e) {
      let src = e.target.result

      context = document.getElementById('uploadedImage').getContext('2d')
      context.clearRect(0, 0, targetSize.w, targetSize.h)

      imageElement = new Image()
      imageElement.src = src

      imageElement.onload = function () {
        let resizeRatio = imageSize / Math.max(imageElement.width, imageElement.height)
        targetSize.w = Math.round(resizeRatio * imageElement.width)
        targetSize.h = Math.round(resizeRatio * imageElement.height)

        let origSize = {
          w: imageElement.width,
          h: imageElement.height
        }
        imageElement.width = targetSize.w
        imageElement.height = targetSize.h

        let canvas = document.getElementById('uploadedImage')
        canvas.width = targetSize.w
        canvas.height = targetSize.w
        canvas
          .getContext('2d')
          .drawImage(imageElement, 0, 0, targetSize.w, targetSize.h)

        message(`resized from ${origSize.w} x ${origSize.h} to ${targetSize.w} x ${targetSize.h}`)
      }
    }

    reader.readAsDataURL(input.files[0])
  } else {
    message('no image uploaded', true)
  }
}

const updateVideoPrediction = async function () {
  const prediction = await runModel(videoCanvas)
  const detected = await processOutput(prediction)

  context.clearRect(0, 0, videoCanvas.width, videoCanvas.height)
  context.drawImage(videoElement, 0, 0, videoCanvas.width, videoCanvas.height)

  renderPredictions(detected)

  requestAnimationFrame(updateVideoPrediction)
}

const updateImagePrediction = async function () {
  context.clearRect(0, 0, imageElement.width, imageElement.height)
  context.drawImage(imageElement, 0, 0, imageElement.width, imageElement.height)

  const prediction = await runModel(videoCanvas)
  const detected = await processOutput(prediction)

  renderPredictions(detected)
}

/**
 * run the model using an image and get a prediction
 */
const runModel = async function (element) {
  // const webcam = document.getElementById('webcamcheckbox').checked
  if (!model) {
    message('model not loaded')
  } else {
    const inputElement = element || imageElement
    if (!inputElement) {
      message('no image available', true)
    } else {
      inputTensor = preprocessInput(inputElement)
      // https://js.tensorflow.org/api/latest/#tf.GraphModel.executeAsync
      const output = await model.executeAsync({
        'image_tensor': inputTensor
      })

      return output
    }
  }
}

/**
 * convert image to Tensor input required by the model
 *
 * @param {HTMLImageElement|HTMLCanvasElement|HTMLVideoElement} imageInput - the image element
 */
function preprocessInput (imageInput) {
  return tf.tidy(() => {
    return tf.browser
      .fromPixels(imageInput)
      .expandDims()
  })
}

/**
 * convert model Tensor output to desired data
 *
 * @param {Tensor} result - the model prediction result
 */
async function processOutput (result) {
  const scores = result[0].dataSync()
  const boxes = result[1].dataSync()

  // clean the webgl tensors
  inputTensor.dispose()
  tf.dispose(result)

  const [maxScores, classes] = calculateMaxScores(
    scores,
    result[0].shape[1],
    result[0].shape[2]
  )

  const boxes2 = tf.tensor2d(boxes, [
    result[1].shape[1],
    result[1].shape[3]
  ])
  const indexTensor = await tf.image.nonMaxSuppressionAsync(
    boxes2,
    maxScores,
    20, // maxNumBoxes
    0.5, // iou_threshold
    0.5 // score_threshold
  )

  const indexes = indexTensor.dataSync()
  boxes2.dispose()
  indexTensor.dispose()

  const height = inputTensor.shape[1]
  const width = inputTensor.shape[2]

  return buildDetectedObjects(
    width,
    height,
    boxes,
    maxScores,
    indexes,
    classes
  )
}

function webcamToggled () {
  const webcam = document.getElementById('webcamcheckbox').checked
  if (webcam) {
    document.body.classList.add('webcam')
  } else {
    stopWebcam()
    document.body.classList.remove('webcam')
  }
  init()
}

const startWebcam = async function () {
  try {
    if (!webcamStream) {
      webcamStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      videoElement = document.createElement('video')
    }

    if (typeof videoElement.srcObject !== 'undefined') {
      videoElement.srcObject = webcamStream
    } else {
      videoElement.src = URL.createObjectURL(webcamStream)
    }

    videoCanvas = document.getElementById('webcamVideo')
    videoCanvas.width = targetSize.w
    videoCanvas.height = targetSize.h
    context = videoCanvas.getContext('2d')

    videoElement.play()
    // wait a sec to give video time to start
    updateVideoPrediction()
  } catch (err) {
    message(err.message || err, true)
  }
}

const stopWebcam = function () {
  if (videoElement && (videoElement.srcObject || videoElement.src)) {
    const videoTracks = webcamStream.getVideoTracks()
    videoTracks.forEach(track => {
      track.stop()
    })

    if (typeof videoElement.srcObject !== 'undefined') {
      videoElement.srcObject = null
    } else {
      videoElement.src = null
    }
  }
  videoElement = null
  webcamStream = null
}

const calculateMaxScores = (scores, numBoxes, numClasses) => {
  const maxes = []
  const classes = []
  for (let i = 0; i < numBoxes; i++) {
    let max = Number.MIN_VALUE
    let index = -1
    for (let j = 0; j < numClasses; j++) {
      if (scores[i * numClasses + j] > max) {
        max = scores[i * numClasses + j]
        index = j
      }
    }
    maxes[i] = max
    classes[i] = index
  }
  return [maxes, classes]
}

const buildDetectedObjects = function (
  width,
  height,
  boxes,
  scores,
  indexes,
  classes
) {
  const count = indexes.length
  const objects = []
  for (let i = 0; i < count; i++) {
    const bbox = []
    for (let j = 0; j < 4; j++) {
      bbox[j] = boxes[indexes[i] * 4 + j]
    }
    const minY = bbox[0] * height
    const minX = bbox[1] * width
    const maxY = bbox[2] * height
    const maxX = bbox[3] * width
    bbox[0] = minX
    bbox[1] = minY
    bbox[2] = maxX - minX
    bbox[3] = maxY - minY
    objects.push({
      bbox: bbox,
      class: classes[indexes[i]],
      score: scores[indexes[i]]
    })
  }
  return objects
}

const renderPredictions = function (predictions) {
  // const context = this.canvasRef.current.getContext('2d')
  // context.clearRect(0, 0, context.canvas.width, context.canvas.height)
  // Font options.
  // console.log('render', predictions.length)
  const font = '16px sans-serif'
  context.font = font
  context.textBaseline = 'top'
  predictions.forEach(prediction => {
    const x = prediction.bbox[0]
    const y = prediction.bbox[1]
    const width = prediction.bbox[2]
    const height = prediction.bbox[3]
    const label = labels[parseInt(prediction.class)]
    // Draw the bounding box.
    context.strokeStyle = '#00FFFF'
    context.lineWidth = 1
    context.strokeRect(x, y, width, height)
    // Draw the label background.
    // context.fillStyle = '#00FFFF'
    // const textWidth = context.measureText(label).width
    // const textHeight = parseInt(font, 10) // base 10
    // context.fillRect(x, y, textWidth + 4, textHeight + 4)
  })

  predictions.forEach(prediction => {
    const x = prediction.bbox[0]
    const y = prediction.bbox[1]
    const height = prediction.bbox[3]
    const label = labels[parseInt(prediction.class)]
    // Draw the text last to ensure it's on top.
    context.fillStyle = '#000000'
    context.fillText(label, x, y - height / 2)
  })
}

function message (msg, highlight) {
  let mark = null
  if (highlight) {
    mark = document.createElement('mark')
    mark.innerText = msg
  }

  const node = document.createElement('div')
  if (mark) {
    node.appendChild(mark)
  } else {
    node.innerText = msg
  }

  document.getElementById('message').appendChild(node)
}

async function init () {
  document.getElementById('message').innerHTML = ''
  message(`tfjs version: ${tf.version.tfjs}`, true)
  await loadModel()
  const res = await fetch(labelsUrl)
  labels = await res.json()
  message(`labels: ${JSON.stringify(labels)}`)
}

// ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init)
} else {
  setTimeout(init, 500)
}
