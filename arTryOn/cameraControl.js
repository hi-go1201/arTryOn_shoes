let width = 0;
let height = 0;

// whether streaming video from the camera.
let streaming = false;

let video = document.getElementById("video");
let stream = null;
let vc = null;

let detectFootArea_flag = false;
let detectFootAreaRect = null;
let x0, x1, y0, y1 = 0.0;

//let info = document.getElementById('info');
//let container = document.getElementById('container');

function startCamera() {
  if (streaming) return;
  navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "environment",
      //width: { min: 800, ideal: 1280, max: 1920 },
      //height: { min: 600, ideal:  720, max: 1080 }
    },
    audio: false
  })
    .then(function(s) {
    stream = s;
    video.srcObject = s;
    video.play();
  })
    .catch(function(err) {
    console.log("An error occured! " + err);
  });

  video.addEventListener("canplay", function(ev){
    if (!streaming) {
      height = video.videoHeight;
      width = video.videoWidth;
      video.setAttribute("width", width);
      video.setAttribute("height", height);
      streaming = true;
      vc = new cv.VideoCapture(video);
    }
    startVideoProcessing();
  }, false);
}

let lastFilter = '';
let src = null;
let dstC1 = null;
let dstC3 = null;
let dstC4 = null;

function startVideoProcessing() {
  if (!streaming) { console.warn("Please startup your webcam"); return; }
  stopVideoProcessing();
  src = new cv.Mat(height, width, cv.CV_8UC4);
  dstC1 = new cv.Mat(height, width, cv.CV_8UC1);
  dstC3 = new cv.Mat(height, width, cv.CV_8UC3);
  dstC4 = new cv.Mat(height, width, cv.CV_8UC4);
  requestAnimationFrame(processVideo);
  addWebGL();
}

function passThrough(src) {
  return src;
}

function gray(src) {
  cv.cvtColor(src, dstC1, cv.COLOR_RGBA2GRAY);
  return dstC1;
}

function hsv(src) {
  cv.cvtColor(src, dstC3, cv.COLOR_RGBA2RGB);
  cv.cvtColor(dstC3, dstC3, cv.COLOR_RGB2HSV);
  return dstC3;
}

function canny(src) {
  cv.cvtColor(src, dstC1, cv.COLOR_RGBA2GRAY);
  cv.Canny(dstC1, dstC1, controls.cannyThreshold1, controls.cannyThreshold2, controls.cannyApertureSize, controls.cannyL2Gradient);
  return dstC1;
}

function inRange(src) {
  let lowValue = controls.inRangeLow;
  let lowScalar = new cv.Scalar(lowValue, lowValue, lowValue, 255);
  let highValue = controls.inRangeHigh;
  let highScalar = new cv.Scalar(highValue, highValue, highValue, 255);
  let low = new cv.Mat(height, width, src.type(), lowScalar);
  let high = new cv.Mat(height, width, src.type(), highScalar);
  cv.inRange(src, low, high, dstC1);
  low.delete(); high.delete();
  return dstC1;
}

function threshold(src) {
  cv.threshold(src, dstC4, controls.thresholdValue, 200, cv.THRESH_BINARY);
  return dstC4;
}

function adaptiveThreshold(src) {
  let mat = new cv.Mat(height, width, cv.CV_8U);
  cv.cvtColor(src, mat, cv.COLOR_RGBA2GRAY);
  cv.adaptiveThreshold(mat, dstC1, 200, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, Number(controls.adaptiveBlockSize), 2);
  mat.delete();
  return dstC1;
}

function gaussianBlur(src) {
  cv.GaussianBlur(src, dstC4, {width: controls.gaussianBlurSize, height: controls.gaussianBlurSize}, 0, 0, cv.BORDER_DEFAULT);
  return dstC4;
}

function bilateralFilter(src) {
  let mat = new cv.Mat(height, width, cv.CV_8UC3);
  cv.cvtColor(src, mat, cv.COLOR_RGBA2RGB);
  cv.bilateralFilter(mat, dstC3, controls.bilateralFilterDiameter, controls.bilateralFilterSigma, controls.bilateralFilterSigma, cv.BORDER_DEFAULT);
  mat.delete();
  return dstC3;
}

function medianBlur(src) {
  cv.medianBlur(src, dstC4, controls.medianBlurSize);
  return dstC4;
}

function sobel(src) {
  var mat = new cv.Mat(height, width, cv.CV_8UC1);
  cv.cvtColor(src, mat, cv.COLOR_RGB2GRAY, 0);
  cv.Sobel(mat, dstC1, cv.CV_8U, 1, 0, controls.sobelSize, 1, 0, cv.BORDER_DEFAULT);
  mat.delete();
  return dstC1;
}

function scharr(src) {
  var mat = new cv.Mat(height, width, cv.CV_8UC1);
  cv.cvtColor(src, mat, cv.COLOR_RGB2GRAY, 0);
  cv.Scharr(mat, dstC1, cv.CV_8U, 1, 0, 1, 0, cv.BORDER_DEFAULT);
  mat.delete();
  return dstC1;
}

function laplacian(src) {
  var mat = new cv.Mat(height, width, cv.CV_8UC1);
  cv.cvtColor(src, mat, cv.COLOR_RGB2GRAY);
  cv.Laplacian(mat, dstC1, cv.CV_8U, controls.laplacianSize, 1, 0, cv.BORDER_DEFAULT);
  mat.delete();
  return dstC1;
}

let contoursColor = [];
for (let i = 0; i < 10000; i++) {
  contoursColor.push([Math.round(Math.random() * 255), Math.round(Math.random() * 255), Math.round(Math.random() * 255), 0]);
}

function contours(src) {
  cv.cvtColor(src, dstC1, cv.COLOR_RGBA2GRAY);
  cv.threshold(dstC1, dstC4, 120, 200, cv.THRESH_BINARY);
  let contours  = new cv.MatVector();
  let hierarchy = new cv.Mat();
  cv.findContours(dstC4, contours, hierarchy, Number(controls.contoursMode), Number(controls.contoursMethod), {x: 0, y: 0});
  dstC3.delete();
  dstC3 = cv.Mat.ones(height, width, cv.CV_8UC3);
  for (let i = 0; i<contours.size(); ++i)
  {
    let color = contoursColor[i];
    cv.drawContours(dstC3, contours, i, color, 1, cv.LINE_8, hierarchy);
  }
  contours.delete(); hierarchy.delete();
  return dstC3;
}

function calcHist(src) {
  cv.cvtColor(src, dstC1, cv.COLOR_RGBA2GRAY);
  let srcVec = new cv.MatVector();
  srcVec.push_back(dstC1);
  let scale = 2;
  let channels = [0], histSize = [src.cols/scale], ranges = [0,255];
  let hist = new cv.Mat(), mask = new cv.Mat(), color = new cv.Scalar(0xfb, 0xca, 0x04, 0xff);
  cv.calcHist(srcVec, channels, mask, hist, histSize, ranges);
  let result = cv.minMaxLoc(hist, mask);
  var max = result.maxVal;
  cv.cvtColor(dstC1, dstC4, cv.COLOR_GRAY2RGBA);
  // draw histogram on src
  for(var i = 0; i < histSize[0]; i++)
  {
      var binVal = hist.data32F[i] * src.rows / max;
      cv.rectangle(dstC4, {x: i * scale, y: src.rows - 1}, {x: (i + 1) * scale - 1, y: src.rows - binVal/3}, color, cv.FILLED);
  }
  srcVec.delete();
  mask.delete();
  hist.delete();
  return dstC4;
}

function equalizeHist(src) {
  cv.cvtColor(src, dstC1, cv.COLOR_RGBA2GRAY, 0);
  cv.equalizeHist(dstC1, dstC1);
  return dstC1;
}

let base;

function backprojection(src) {
  if (lastFilter !== 'backprojection') {
    if (base instanceof cv.Mat)
      base.delete();
    base = src.clone();
    cv.cvtColor(base, base, cv.COLOR_RGB2HSV, 0);
  }
  cv.cvtColor(src, dstC3, cv.COLOR_RGB2HSV, 0);
  let baseVec = new cv.MatVector(), targetVec = new cv.MatVector();
  baseVec.push_back(base); targetVec.push_back(dstC3);
  let mask = new cv.Mat(), hist = new cv.Mat();
  let channels = [0], histSize = [50];
  let ranges;
  if (controls.backprojectionRangeLow < controls.backprojectionRangeHigh)
    ranges = [controls.backprojectionRangeLow, controls.backprojectionRangeHigh];
  else
    return src;
  cv.calcHist(baseVec, channels, mask, hist, histSize, ranges);
  cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX);
  cv.calcBackProject(targetVec, channels, hist, dstC1, ranges, 1);
  baseVec.delete();
  targetVec.delete();
  mask.delete();
  hist.delete();
  return dstC1;
}

function erosion(src) {
  let kernelSize = controls.erosionSize;
  let kernel = cv.Mat.ones(kernelSize, kernelSize, cv.CV_8U);
  let color = new cv.Scalar();
  cv.erode(src, dstC4, kernel, {x: -1, y: -1}, 1, Number(controls.erosionBorderType), color);
  kernel.delete();
  return dstC4;
}

function dilation(src) {
  let kernelSize = controls.dilationSize;
  let kernel = cv.Mat.ones(kernelSize, kernelSize, cv.CV_8U);
  let color = new cv.Scalar();
  cv.dilate(src, dstC4, kernel, {x: -1, y: -1}, 1, Number(controls.dilationBorderType), color);
  kernel.delete();
  return dstC4;
}

function morphology(src) {
  let kernelSize = controls.morphologySize;
  let kernel = cv.getStructuringElement(Number(controls.morphologyShape), {width: kernelSize, height: kernelSize});
  let color = new cv.Scalar();
  let op = Number(controls.morphologyOp);
  let image = src;
  if (op === cv.MORPH_GRADIENT || op === cv.MORPH_TOPHAT || op === cv.MORPH_BLACKHAT) {
    cv.cvtColor(src, dstC3, cv.COLOR_RGBA2RGB);
    image = dstC3;
  }
  cv.morphologyEx(image, dstC4, op, kernel, {x: -1, y: -1}, 1, Number(controls.morphologyBorderType), color);
  kernel.delete();
  return dstC4;
}

function processVideo() {
  //stats.begin();
  vc.read(src);
  let result;
  switch (controls.filter) {
    case 'passThrough': result = passThrough(src); break;
    case 'gray': result = gray(src); break;
    case 'hsv': result = hsv(src); break;
    case 'canny': result = canny(src); break;
    case 'inRange': result = inRange(src); break;
    case 'threshold': result = threshold(src); break;
    case 'adaptiveThreshold': result = adaptiveThreshold(src); break;
    case 'gaussianBlur': result = gaussianBlur(src); break;
    case 'bilateralFilter': result = bilateralFilter(src); break;
    case 'medianBlur': result = medianBlur(src); break;
    case 'sobel': result = sobel(src); break;
    case 'scharr': result = scharr(src); break;
    case 'laplacian': result = laplacian(src); break;
    case 'contours': result = contours(src); break;
    case 'calcHist': result = calcHist(src); break;
    case 'equalizeHist': result = equalizeHist(src); break;
    case 'backprojection': result = backprojection(src); break;
    case 'erosion': result = erosion(src); break;
    case 'dilation': result = dilation(src); break;
    case 'morphology': result = morphology(src); break;
    case 'detectFootArea': result = detectFootArea(src); break;
    default: result = passThrough(src);
  }
  cv.imshow("canvas", result);
  //stats.end();
  lastFilter = controls.filter;
  requestAnimationFrame(processVideo);
}

function stopVideoProcessing() {
  if (src != null && !src.isDeleted()) src.delete();
  if (dstC1 != null && !dstC1.isDeleted()) dstC1.delete();
  if (dstC3 != null && !dstC3.isDeleted()) dstC3.delete();
  if (dstC4 != null && !dstC4.isDeleted()) dstC4.delete();
}

function stopCamera() {
  if (!streaming) return;
  stopVideoProcessing();
  video.pause();
  video.srcObject=null;
  stream.getVideoTracks()[0].stop();
  streaming = false;
}

var stats = null;

var filters = {
  'passThrough': 'Pass Through',
  'gray': 'Gray',
  'hsv': 'HSV',
  'canny': 'Canny Edge Detection',
  'inRange': 'In Range',
  'threshold': 'Threshold',
  'adaptiveThreshold': 'Adaptive Threshold',
  'gaussianBlur': 'Gaussian Blurring',
  'medianBlur': 'Median Blurring',
  'bilateralFilter': 'Bilateral Filtering',
  'sobel': 'Sobel Derivatives',
  'scharr': 'Scharr Derivatives',
  'laplacian': 'Laplacian Derivatives',
  'contours': 'Contours',
  'calcHist': 'Calculation',
  'equalizeHist': 'Equalization',
  'backprojection': 'Backprojection',
  'erosion': 'Erosion',
  'dilation': 'Dilation',
  'morphology': 'Morphology',
};

var filterName = document.getElementById('filterName');

var controls;

function initUI() {
  //stats = new Stats();
  //stats.showPanel(0);
  //document.getElementById('container').appendChild(stats.domElement);

  controls = {
    filter: 'detectFootArea',
    setFilter: function(filter) {
      this.filter = filter;
      filterName.innerHTML = filters[filter];
    },
    passThrough: function() { this.setFilter('passThrough'); },
    gray: function() { this.setFilter('gray'); },
    hsv: function() { this.setFilter('hsv'); },
    inRange: function() { this.setFilter('inRange'); },
    inRangeLow: 75,
    inRangeHigh: 150,
    threshold: function() { this.setFilter('threshold'); },
    thresholdValue: 100,
    adaptiveThreshold: function() { this.setFilter('adaptiveThreshold'); },
    adaptiveBlockSize: 3,
    gaussianBlur: function() { this.setFilter('gaussianBlur'); },
    gaussianBlurSize: 7,
    medianBlur: function() { this.setFilter('medianBlur'); },
    medianBlurSize: 5,
    bilateralFilter: function() { this.setFilter('bilateralFilter'); },
    bilateralFilterDiameter: 5,
    bilateralFilterSigma: 75,
    sobel: function() { this.setFilter('sobel'); },
    sobelSize: 3,
    scharr: function() { this.setFilter('scharr'); },
    laplacian: function() { this.setFilter('laplacian'); },
    laplacianSize: 3,
    canny: function() { this.setFilter('canny'); },
    cannyThreshold1: 150,
    cannyThreshold2: 300,
    cannyApertureSize: 3,
    cannyL2Gradient: false,
    contours: function() { this.setFilter('contours'); },
    contoursMode: cv.RETR_CCOMP,
    contoursMethod: cv.CHAIN_APPROX_SIMPLE,
    calcHist: function() { this.setFilter('calcHist'); },
    equalizeHist: function() { this.setFilter('equalizeHist'); },
    backprojection: function() { this.setFilter('backprojection'); },
    backprojectionRangeLow: 0,
    backprojectionRangeHigh: 150,
    morphology: function () { this.setFilter('morphology'); },
    morphologyShape: cv.MORPH_RECT,
    morphologyOp: cv.MORPH_ERODE,
    morphologySize: 5,
    morphologyBorderType: cv.BORDER_CONSTANT,
  };
  /*
  let gui = new dat.GUI({ autoPlace: false });
  let guiContainer = document.getElementById('guiContainer');
  guiContainer.appendChild(gui.domElement);
  */
  let lastFolder = null;
  function closeLastFolder(folder) {
    if (lastFolder != null && lastFolder != folder) {
      lastFolder.close();
    }
    lastFolder = folder;
  }
  /*
  let passThrough = gui.add(controls, 'passThrough').name(filters['passThrough']).onChange(function() {
    closeLastFolder(null);
  });
  
  let colorConversion = gui.addFolder('Color Conversion');
  colorConversion.add(controls, 'gray').name(filters['gray']).onChange(function() {
    closeLastFolder(null);
  });
  
  colorConversion.add(controls, 'hsv').name(filters['hsv']).onChange(function() {
    closeLastFolder(null);
  });
  
  let inRange = colorConversion.addFolder(filters['inRange']);
  inRange.domElement.onclick = function() {
    closeLastFolder(inRange);
    controls.inRange();
  };
  inRange.add(controls, 'inRangeLow', 0, 255, 1).name('lower boundary');
  inRange.add(controls, 'inRangeHigh', 0, 255, 1).name('higher boundary');
  
  // let geometricTransformations = gui.addFolder('Geometric Transformations');
  // TODO
  
  let thresholding = gui.addFolder('Thresholding');
  
  let threshold = thresholding.addFolder(filters['threshold']);
  threshold.domElement.onclick = function() {
    closeLastFolder(threshold);
    controls.threshold();
  };
  threshold.add(controls, 'thresholdValue', 0, 200, 1).name('threshold value');
  
  let adaptiveThreshold = thresholding.addFolder(filters['adaptiveThreshold']);
  adaptiveThreshold.domElement.onclick = function() {
    closeLastFolder(adaptiveThreshold);
    controls.adaptiveThreshold();
  };
  adaptiveThreshold.add(controls, 'adaptiveBlockSize', 3, 99, 1).name('block size').onChange(function(value) { if (value % 2 === 0) controls.adaptiveBlockSize = value + 1;});
  
  let smoothing = gui.addFolder('Smoothing');
  
  let gaussianBlur = smoothing.addFolder(filters['gaussianBlur']);
  gaussianBlur.domElement.onclick = function() {
    closeLastFolder(gaussianBlur);
    controls.gaussianBlur();
  };
  gaussianBlur.add(controls, 'gaussianBlurSize', 7, 99, 1).name('kernel size').onChange(function(value) { if (value % 2 === 0) controls.gaussianBlurSize = value + 1;});
  
  let medianBlur = smoothing.addFolder(filters['medianBlur']);
  medianBlur.domElement.onclick = function() {
    closeLastFolder(medianBlur);
    controls.medianBlur();
  };
  medianBlur.add(controls, 'medianBlurSize', 3, 99, 1).name('kernel size').onChange(function(value) { if (value % 2 === 0) controls.medianBlurSize = value + 1;});
  
  let bilateralFilter = smoothing.addFolder(filters['bilateralFilter']);
  bilateralFilter.domElement.onclick = function() {
    closeLastFolder(bilateralFilter);
    controls.bilateralFilter();
  };
  bilateralFilter.add(controls, 'bilateralFilterDiameter', 1, 15, 1).name('diameter');
  bilateralFilter.add(controls, 'bilateralFilterSigma', 1, 255, 1).name('sigma')
  
  let morphology = gui.addFolder('Morphology');
  morphology.domElement.onclick = function() {
    closeLastFolder(morphology);
    controls.morphology();
  };
  morphology.add(controls, 'morphologyOp', {'MORPH_ERODE': cv.MORPH_ERODE, 'MORPH_DILATE': cv.MORPH_DILATE, 'MORPH_OPEN ': cv.MORPH_OPEN, 'MORPH_CLOSE': cv.MORPH_CLOSE, 'MORPH_GRADIENT': cv.MORPH_GRADIENT, 'MORPH_TOPHAT': cv.MORPH_TOPHAT, 'MORPH_BLACKHAT': cv.MORPH_BLACKHAT}).name('operation');
  morphology.add(controls, 'morphologyShape', {'MORPH_RECT': cv.MORPH_RECT, 'MORPH_CROSS': cv.MORPH_CROSS, 'MORPH_ELLIPSE': cv.MORPH_ELLIPSE}).name('shape');
  morphology.add(controls, 'morphologySize', 1, 15, 1).name('kernel size').onChange(function(value) { if (value % 2 === 0) controls.morphologySize = value + 1;});
  morphology.add(controls, 'morphologyBorderType', {'BORDER_CONSTANT': cv.BORDER_CONSTANT, 'BORDER_REPLICATE': cv.BORDER_REPLICATE, 'BORDER_REFLECT': cv.BORDER_REFLECT, 'BORDER_REFLECT_101': cv.BORDER_REFLECT_101}).name('boarder type');

  let gradients = gui.addFolder('Gradients')
  let sobel = gradients.addFolder(filters['sobel']);
  sobel.domElement.onclick = function() {
    closeLastFolder(sobel);
    controls.sobel();
  };
  sobel.add(controls, 'sobelSize', 3, 19, 1).name('kernel size').onChange(function(value) { if (value % 2 === 0) controls.sobelSize = value + 1;});
  
  gradients.add(controls, 'scharr').name(filters['scharr']).onChange(function() {
    closeLastFolder(null);
  });

  let laplacian = gradients.addFolder(filters['laplacian']);
  laplacian.domElement.onclick = function() {
    closeLastFolder(laplacian);
    controls.laplacian();
  };
  laplacian.add(controls, 'laplacianSize', 1, 19, 1).name('kernel size').onChange(function(value) { if (value % 2 === 0) controls.laplacianSize = value + 1;});

  let canny = gui.addFolder(filters['canny']);
  canny.domElement.onclick = function() {
    closeLastFolder(canny);
    controls.canny();
  };
  canny.add(controls, 'cannyThreshold1', 1, 500, 1).name('threshold1');
  canny.add(controls, 'cannyThreshold2', 1, 500, 1).name('threshold2');
  canny.add(controls, 'cannyApertureSize', 3, 7, 1).name('aperture size').onChange(function(value) { if (value % 2 === 0) controls.cannyApertureSize = value + 1;});
  canny.add(controls, 'cannyL2Gradient').name('l2 gradient');

  let contours = gui.addFolder(filters['contours']);
  contours.domElement.onclick = function() {
    closeLastFolder(contours);
    controls.contours();
  };
  contours.add(controls, 'contoursMode', {'RETR_EXTERNAL': cv.RETR_EXTERNAL, 'RETR_LIST': cv.RETR_LIST, 'RETR_CCOMP': cv.RETR_CCOMP, 'RETR_TREE': cv.RETR_TREE}).name('mode');
  contours.add(controls, 'contoursMethod', {'CHAIN_APPROX_NONE': cv.CHAIN_APPROX_NONE, 'CHAIN_APPROX_SIMPLE': cv.CHAIN_APPROX_SIMPLE, 'CHAIN_APPROX_TC89_L1': cv.CHAIN_APPROX_TC89_L1, 'CHAIN_APPROX_TC89_KCOS': cv.CHAIN_APPROX_TC89_KCOS}).name('method');
  
  let histograms = gui.addFolder('Histograms');
  histograms.add(controls, 'calcHist').name(filters['calcHist']).onChange(function() {
    closeLastFolder(null);
  })
  histograms.add(controls, 'equalizeHist').name(filters['equalizeHist']).onChange(function() {
    closeLastFolder(null);
  });
  
  let backprojection = histograms.addFolder(filters['backprojection']);
  backprojection.domElement.onclick = function() {
    closeLastFolder(backprojection);
    controls.backprojection();
  };
  backprojection.add(controls, 'backprojectionRangeLow', 0, 255, 1).name('range low');
  backprojection.add(controls, 'backprojectionRangeHigh', 0, 255, 1).name('range high');
*/
}

function opencvIsReady() {
  console.log('OpenCV.js is ready');
  //info.innerHTML = '';
  //container.className = '';
  initUI();
  startCamera();
}

function detectFootArea(src) {
  let mat = new cv.Mat(src.rows, src.cols, cv.CV_8UC4);
  //let dst = cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC3);
  /*靴下領域を抽出する処理*/
  cv.GaussianBlur(src, mat, {width: 7, height: 7}, 0, 0, cv.BORDER_DEFAULT);
  cv.cvtColor(mat, mat, cv.COLOR_RGBA2GRAY, 0);
  cv.adaptiveThreshold(mat, mat, 200, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 3, 2);
  let contours  = new cv.MatVector();
  let hierarchy = new cv.Mat();
  //let poly = new cv.MatVector();
  //let hull = new cv.MatVector();
  cv.findContours(mat, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
  // approximates each contour to polygon
  //console.log("detectFootArea Start");
  var cnt_detectFootArea = 0;
  for (let i = 0; i < contours.size(); ++i) {
    let tmp = new cv.Mat();
    let cnt = contours.get(i);
    // You can try more different parameters
    //cv.approxPolyDP(cnt, tmp, 3, true);
    //poly.push_back(tmp);
    //cv.convexHull(cnt, tmp, false, true);
    //hull.push_back(tmp);
    let rotatedRect = cv.minAreaRect(cnt);
    //console.log(rotatedRect.size.width);
    if( (rotatedRect.size.height > rotatedRect.size.width*2)
        && rotatedRect.size.width > 100
        && rotatedRect.size.height > 100
        && rotatedRect.size.width < width
        && rotatedRect.size.height < height
      ) {
      let vertices = cv.RotatedRect.points(rotatedRect);
      
      detectFootAreaRect = rotatedRect;
      //console.log(rotatedRect.angle);
      //console.log(rotatedRect);
      
      x0 = rotatedRect.center.x - rotatedRect.size.width*0.5;
      x1 = rotatedRect.center.x + rotatedRect.size.width*0.5;
      y0 = rotatedRect.center.y - rotatedRect.size.height*0.5;
      y1 = rotatedRect.center.y + rotatedRect.size.height*0.5;

      //console.log("rotatedRect.pos.x0:" + x0);
      //console.log("rotatedRect.pos.y0:" + y0);
      //console.log("rotatedRect.pos.x1:" + x1);
      //console.log("rotatedRect.pos.y1:" + y1);
      //console.log("rotatedRect.size.width:" + rotatedRect.size.width);
      //console.log("rotatedRect.size.height:" + rotatedRect.size.height);
      //console.log("rotatedRect.angle:" + rotatedRect.angle);
      
      //console.log("rotatedRect.size.width:" + rotatedRect.size.width);
      //console.log("rotatedRect.size.height:" + rotatedRect.size.height);

      let contoursColor = new cv.Scalar(255, 255, 255);
      let rectangleColor = new cv.Scalar(255, 0, 0, 255);
      //cv.drawContours(src, contours, 0, contoursColor, 1, 8, hierarchy, 100);
      // draw rotatedRect
      for (let i = 0; i < 4; i++) {
        cv.line(src, vertices[i], vertices[(i + 1) % 4], rectangleColor, 2, cv.LINE_AA, 0);
      }
      cnt_detectFootArea += 1;
    }
      /*
      let rect = cv.boundingRect(cnt);
      let contoursColor = new cv.Scalar(255, 255, 255);
      let rectangleColor = new cv.Scalar(255, 0, 0);
      let color = new cv.Scalar(Math.round(Math.random() * 255), Math.round(Math.random() * 255),
                                Math.round(Math.random() * 255));
      cv.drawContours(dst, contours, i, color, 1, 8, hierarchy, 100);
      let point1 = new cv.Point(rect.x, rect.y);
      let point2 = new cv.Point(rect.x + rect.width, rect.y + rect.height);
      cv.rectangle(dst, point1, point2, rectangleColor, 2, cv.LINE_AA, 0);
      */
    cnt.delete(); tmp.delete();
  }
  if(cnt_detectFootArea > 0){
    detectFootArea_flag = true;
  } else {
    detectFootArea_flag = false;
  }
    /*
    console.log(contours.size());
    let contoursColor = [];
    for (let i = 0; i < 10000; i++) {
      contoursColor.push([Math.round(Math.random() * 255), Math.round(Math.random() * 255), Math.round(Math.random() * 255), 0]);
    }
    for (let i = 0; i<contours.size(); ++i)
    {
      let color = contoursColor[i];
      console.log(color);
      cv.drawContours(src, contours, i, color, 5, cv.LINE_8, hierarchy);
    }
    contours.delete(); hierarchy.delete();
    */
  mat.delete(); contours.delete(); hierarchy.delete();
  //src.delete(); dst.delete(); contours.delete(); hierarchy.delete(); poly.delete(); hull.delete();
  return src;
}

function addWebGL() {
  // init
  var width = window.innerWidth;
  var height = window.innerHeight;

  // Stats
  const stats = new Stats();
  stats.setMode(0);
  stats.domElement.style.position = "absolute";
  stats.domElement.style.left = "0px";
  stats.domElement.style.top  = "0px";
  document.body.appendChild(stats.dom);

  // Set up the main camera
  const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
  camera.position.z = 5;

  const scene = new THREE.Scene();

  // Create lights
  var light = new THREE.PointLight(0xEEEEEE);
  light.position.set(20, 0, 20);
  scene.add(light);
  var lightAmb = new THREE.AmbientLight(0x777777);
  scene.add(lightAmb);
  // 平行光源
  var lightDir = new THREE.DirectionalLight(0xFFFFFF);
  lightDir.intensity = 5; // 光の強さを倍に
  lightDir.position.set(1, 1, 1);
  scene.add(lightDir);

  //ここで3Dモデルをロード
  //今回はglTF形式のものを使用
  var model1 = null;//左足
  var model2 = null;//右足

  const loader = new THREE.GLTFLoader();
  loader.load('./obj/BasketballShoe.glb', 
    function (gltf) {
      model1 = gltf.scene; // THREE.Group
      model1.name = "shoes1"
      model1.visible = false;
      model1.scale.set(0.3, 0.3, 0.3);
      model1.position.set(-0.5, 0.0, 0.0);
      model1.rotation.x = -1.4;
      model1.rotation.y = 0;
      model1.rotation.z = 3.3;
      scene.add(model1);
    },
    // called while loading is progressing
    function (xhr) {
      console.log('shoes1: ' + (xhr.loaded / xhr.total * 100) + '% loaded');
    },
    // called when loading has errors
    function (error) {
      console.log('An error happened');
    }
  );
  
  loader.load('./obj/shoes.glb',
    function (gltf) {
      model2 = gltf.scene; // THREE.Group
      model2.name = "shoes2"
      model2.visible = false;
      model2.scale.set(1.0, 1.0, 1.0);
      model2.position.set(0.5, 0.0, 0.0);
      model2.rotation.x = -1.4;
      model2.rotation.y = 0;
      model2.rotation.z = 3.3;
      scene.add(model2);
    },
    // called while loading is progressing
    function (xhr) {
      console.log('shoes2: ' + (xhr.loaded / xhr.total * 100 ) + '% loaded');
    },
    // called when loading has errors
    function (error) {
      console.log('An error happened');
    }
  );

  // renderer
  var renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true
  });
  renderer.setPixelRatio( window.devicePixelRatio );
  renderer.setSize( window.innerWidth, window.innerHeight );
  renderer.autoClear = false; // To allow render overlay on top of sprited sphere

  //document.body.appendChild( renderer.domElement );
  document.getElementById("main").appendChild(renderer.domElement);
  renderer.domElement.id = "webgl";
  // カメラ制御
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0);
  controls.update();

  window.addEventListener( 'resize', onWindowResize, false );

  requestAnimationFrame(render);



  function onWindowResize() {

    var resize_width = window.innerWidth;
    var resize_height = window.innerHeight;

    camera.aspect = resize_width / resize_height;
    camera.updateProjectionMatrix();
  
    renderer.setSize( window.innerWidth, window.innerHeight );

  }

  function render(time) {

    time *= 0.001;

    // create camera image
    var texture = new THREE.Texture(document.querySelector('#canvas'));
    texture.needsUpdate = true; 
    scene.background = texture;
    // Set the repeat and offset properties of the background texture
    // to keep the image's aspect correct.
    // Note the image may not have loaded yet.
    const canvasAspect = window.innerWidth / window.innerHeight;
    const imageAspect = texture.image ? texture.image.width / texture.image.height : 1;
    const aspect = imageAspect / canvasAspect;

    texture.offset.x = aspect > 1 ? (1 - 1 / aspect) / 2 : 0;
    texture.repeat.x = aspect > 1 ? 1 / aspect : 1;

    texture.offset.y = aspect > 1 ? 0 : (1 - aspect) / 2;
    texture.repeat.y = aspect > 1 ? 1 : aspect;

    //sprite sample
    //var material = new THREE.SpriteMaterial( { map: texture } );
    //var width = material.map.image.width;
    //var height = material.map.image.height;
    //sprite = new THREE.Sprite( material );
    //sprite.center.set(0.5, 0.5);
    //sprite.scale.set(width, height, 1);
    //sprite.position.set(0, 0, 1); // center

    
    if(model1 != null && model2 != null){
      //console.log(model);
      //model1.visible = false;
      //console.log(detectFootArea_flag);
      if(detectFootArea_flag == true){
        //スクリーン座標逆変換
        let detectFootArea_sx = detectFootAreaRect.center.x;
        let detectFootArea_sy = detectFootAreaRect.center.y;
        let project_x = (detectFootArea_sx * 2 / width) -1.0 -texture.offset.x;
        let project_y = -(detectFootArea_sy * 2 / height) +1.0 +texture.offset.y;
        console.log(project_x);
        console.log(project_y);
        model1.position.set(project_x, project_y, 0.0);
        //足の抽出角度に応じて3Dモデル回転
        console.log(THREE.Math.degToRad(detectFootAreaRect.angle));
        model1.rotation.y = THREE.Math.degToRad(detectFootAreaRect.angle);
        
        // ToDo:スマホのセンサ情報用いてスマホの傾きに応じて3Dモデルの奥行きの角度調整
        
        model1.visible = true;
        //model2.visible = true;
      } else if(detectFootArea_flag == false){
        model1.visible = false;
        //model2.visible = false;
      }
    
      // スクリーン座標を取得する
      //const project = model1.position.project(camera);
      //const sx = (width / 2) * (+project.x + 1.0);
      //const sy = (height / 2) * (-project.y + 1.0);
      // スクリーン座標
      //console.log("screen pos:" + sx, sy);
      //model1.position.set(-0.5, 0.0, 0.0);

    
      // ワールド座標を取得する
      //const world = model1.getWorldPosition();
      // ワールド座標
      //console.log(world);

    }
    

    stats.update(); // 毎フレームごとにstats.update()を呼ぶ必要がある。

    renderer.clear();
    renderer.clearDepth();
    renderer.render( scene, camera );
    requestAnimationFrame(render);
  }
  
}