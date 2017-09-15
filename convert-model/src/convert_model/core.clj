(ns convert-model.core
  (:import  (org.deeplearning4j.nn.modelimport.keras KerasModelImport)
            (org.deeplearning4j.util                 ModelSerializer)))

(defn model
  []
  (KerasModelImport/importKerasSequentialModelAndWeights "../create-model/model.json" "../create-model/model-weights.h5"))

(defn convert
  []
  (ModelSerializer/writeModel (model)  "../consume-model/app/src/main/res/raw/model.zip" false))
