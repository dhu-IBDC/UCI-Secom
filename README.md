# UCI-Secom
A data mining project based on maching learning and evolutionary computing approaches using UCI Secom data set

UCI-Secom is a data set available on UCI Machine Learning Reposery for researchers in industrial field who are interested in solving practical manufacturing problems via maching learning methodologies. It contains data from a semi-conductor manufacturing process which were collected and organized by Michael McCann and Adrian Johnston. Secom data set can be reached by the following site:

http://archive.ics.uci.edu/ml/datasets/secom

As a special issue in manufacturing research field, the defect detection of wafer products has always been an eye catcher in academia and challenge in daily production. In order to improve the accuracy of defect detection and free manpower from heavy labouring, a complex modern semi-conductor manufacturing process is normally under consistent surveillance via the monitoring of signals/variables collected from sensors and or process measurement points, providing sufficient samples to exvacate patterns from the past. However, not all of these signals are equally valuable in a specific monitoring system. The measured signals contain a combination of useful information, irrelevant information as well as noise. Meanwhile, defective wafers are rare compared to the qualified ones in practice. As a result, th data set shows an obvious characteristic that the ratio between different classes are imbalanced which may lead to severe bias of the classifier.

Hence, three different scientific problems are hidden inside this data set, which are the Feature Selection problem, the Imbalanced Learning problem and the Classification problem. Before implementing specific solutions to solve these problems, raw data also needs cleansing to filter those 'Nan' and blanks in the data set. After the data preprocessing process, different solutions towards three scientific problems can be choosed and optimized. Finally, by combining those locally optimized methods with different ordering stategies, the accuracy of fault detection can be optimized as a whole.

The detailed attributes of the data set is shown as follows:

  Number of Instances:1567
  
  Number of Attributes:591
  
  Missing Values?:Yes
  
  Attribute Characteristics:Real
  
  Number of Classes:2
  
  Imbalanced ratio: approximately 15:1
  
  Associated Tasks:Data Cleansing, Feature Selection, Classification, Imbalaced Learning

In this project, four different complete soluctions designed by four different groups of students from Donghua University China are presented. The main object of the project is to achieves over 72% of recall accuracy (which is defines by the number of correctly-detected  products divides the total number of defect products). The result of those solutions is shown as four pdf files in four different folders and the programs used is also affiliated in it. We are looking forward to hearing from your kindly suggestions toward any step in our project. 
