using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace DifferentialEvolution
{
	class DataSet
	{
		//Attributes
		private string name;
		private int size;
		private int dimensionality;
		private int noClasses;
		private double[][] data;
		private int[] dataLabels;
		private string[] labels;
		private double[] maxFeatureValues;
		private double[] minFeatureValues;

		//Constructors
		public DataSet(int noClasses, string[] classLabels)
		{
			name = null;
			size = 0;
			dimensionality = 0;
			data = null;
			maxFeatureValues = null;
			minFeatureValues = null;

			this.noClasses = noClasses;
			labels = new string[noClasses];
			Array.Copy(classLabels, labels, noClasses);
		}

		public DataSet(string dataFile)
		{
			ImportData(dataFile);
		}

		//Methods
		public bool ImportData(string dataFile)
		{
			StreamReader reader;
			string rawData;
			string[] splitData;
			List<string> tempData = new List<string>();

			name = dataFile;
			try
			{
				reader = new StreamReader(dataFile);
				while ((rawData = reader.ReadLine()) != null)
					tempData.Add(rawData);
				splitData = tempData[0].Split(' ', '\t');

				size = tempData.Count;
				dimensionality = splitData.Length - 1;
				data = new double[size][];
				for (int i = 0; i < size; i++)
					data[i] = new double[dimensionality];
				maxFeatureValues = new double[dimensionality];
				minFeatureValues = new double[dimensionality];

				dataLabels = new int[size];

				for (int i = 0; i < size; i++)
				{
					splitData = tempData[i].Split(' ', '\t');
					for (int j = 0; j < dimensionality; j++)
					{
						data[i][j] = double.Parse(splitData[j]);
						if (i == 0 || data[i][j] > maxFeatureValues[j])
							maxFeatureValues[j] = data[i][j];
						if (i == 0 || data[i][j] < minFeatureValues[j])
							minFeatureValues[j] = data[i][j];
					}
					for (int j = 0; j < labels.Length; j++)
					{
						if (splitData[splitData.Length - 1] == labels[j])
						{
							dataLabels[i] = j;
						}
					}
				}

			}
			catch (Exception e)
			{
				Console.WriteLine("ERROR!");
				Console.WriteLine(e.Message);
				return false;
			}

			return true;
		}

		public double GetMaxFeatureValue(int i) { return maxFeatureValues[i]; }
		public double GetMinFeatureValue(int i) { return minFeatureValues[i]; }

		//Indexer
		public double[] this[int i]
		{
			get { return data[i]; }
		}

		//Properties
		public string Name { get { return name; } }
		public int Size { get { return size; } }
		public int Dimensionality { get { return dimensionality; } }
		public int Classes { get { return noClasses; } }
		public double[][] Data { get { return data; } }
		public int[] DataLables { get { return dataLabels; } }

		public DataSet[] splitDataset(Random rnd, double[] perc)
		{
			int dn = perc.Length;
			DataSet[] Datasets = new DataSet[dn];
			int[] setInstances = new int[dn];
			int totalInstances = this.Size;
			int nClasses = this.Classes;

			List<double[]>[] dataPerClass = new List<double[]>[nClasses];
			for (int i = 0; i < nClasses; i++)
			{
				dataPerClass[i] = new List<double[]>();
			}

			for (int i = 0; i < totalInstances; i++)
			{
				dataPerClass[DataLables[i]].Add(data[i]);
			}

			int[][] setClassCounts = new int[dn][];
			for (int i = 0; i < dn; i++)
			{
				setClassCounts[i] = new int[nClasses];
				for (int j = 0; j < nClasses; j++)
				{
					setClassCounts[i][j] = (int)(dataPerClass[j].Count * perc[i] / 100);
				}
			}
			List<double[]>[] dataPerSet = new List<double[]>[dn];
			List<int>[] labelsPerSet = new List<int>[dn];

			for (int i = 0; i < dn; i++)
			{
				dataPerSet[i] = new List<double[]>();
				labelsPerSet[i] = new List<int>();
				for (int j = 0; j < nClasses; j++)
				{
					for (int k = 0; k < setClassCounts[i][j]; k++)
					{
						int idx = rnd.Next(0, dataPerClass[j].Count);
						dataPerSet[i].Add(dataPerClass[j][idx]);
						labelsPerSet[i].Add(j);
						dataPerClass[j].RemoveAt(idx);
					}
				}

				if (i == dn - 1)
				{
					for (int j = 0; j < nClasses; j++)
					{
						while (dataPerClass[j].Count > 0)
						{
							dataPerSet[i].Add(dataPerClass[j][0]);
							labelsPerSet[i].Add(j);
							dataPerClass[j].RemoveAt(0);
						}
					}
				}

				Datasets[i] = new DataSet(dataPerSet[i], labelsPerSet[i], nClasses, "set" + i);
			}
			return Datasets;
		}

		//KONSTRUKTOR ZA SPLITANJE
		private DataSet(List<double[]> data, List<int> labels, int noClasses, string name)
		{
			this.dimensionality = data[0].Length;
			this.size = data.Count;
			this.noClasses = noClasses;
			this.name = name;

			this.data = new double[size][];
			this.dataLabels = new int[size];
			for (int i = 0; i < size; i++)
			{
				this.data[i] = new double[dimensionality];
				Array.Copy(data[i], this.data[i], dimensionality);
				dataLabels[i] = labels[i];
			}
		}

		public void Save(string fileName)
		{
			StreamWriter sw = new StreamWriter(fileName);
			for (int i = 0; i < size; i++)
			{
				for (int j = 0; j < dimensionality; j++)
					sw.Write(data[i][j] + "\t");
				sw.Write(dataLabels[i] + 1);

				if (i < size - 1)
					sw.Write(Environment.NewLine);
			}

			sw.Close();
		}

		public static DataSet operator +(DataSet d1, DataSet d2)
		{
			List<double[]> d = new List<double[]>();
			List<int> dl = new List<int>();
			for (int i = 0; i < d1.size; i++)
			{
				d.Add(d1[i]);
				dl.Add(d1.dataLabels[i]);
			}
			for (int i = 0; i < d2.size; i++)
			{
				d.Add(d2[i]);
				dl.Add(d2.dataLabels[i]);
			}

			DataSet D = new DataSet(d, dl, d1.noClasses, "union");

			return D;
		}
	}
}
