using System;
using System.Collections.Generic;
using System.Text;

namespace kmeansrbfnn
{
	static class Utilities
	{
		public static double[] getWidts(double[][] centers)
		{
			int k = centers.Length;
			double[] widths = new double[k];

			double dmax = 0, tmp;
			for (int i = 0; i < k; i++)
				for (int j = 0; j < k; j++)
					if (i != j)
					{
						tmp = L2norm(centers[i], centers[j]);
						if (i == 0 || dmax < tmp)
							dmax = tmp;
					}

			double w = dmax / Math.Sqrt(2 * k);
			for (int i = 0; i < k; i++)
				widths[i] = w;

			return widths;
		}

		private static double L2norm(double[] v1, double[] v2)
		{
			double result = 0;
			for (int i = 0; i < v1.Length; i++)
			{
				result += (v1[i] - v2[i]) * (v1[i] - v2[i]);
			}
			return Math.Sqrt(result);
		}
	}
}
