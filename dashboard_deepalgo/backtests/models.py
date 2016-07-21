from django.db import models

class Cohort(models.Model):
    train_start = models.DateField()
    train_end= models.DateField()
    test_start = models.DateField()
    test_end = models.DateField()
    model_type = models.CharField(max_length = 40)
    model_description = models.CharField(max_length = 200)
    #train_iterations, train_return, test_return, 
    def __str__(self):
        return self.model_type

class Strat(models.Model):
    pub_date = models.DateTimeField('date_published')
    train_return = models.FloatField()
    test_return = models.FloatField()
    cohort = models.ForeignKey(Cohort,  on_delete=models.CASCADE)
