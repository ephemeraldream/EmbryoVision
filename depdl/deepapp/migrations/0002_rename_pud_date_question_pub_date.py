# Generated by Django 3.2.23 on 2024-01-29 05:14

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('deepapp', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='question',
            old_name='pud_date',
            new_name='pub_date',
        ),
    ]
